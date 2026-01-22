"""AlphaZero-style training script for TicTacToe, Connect4, and ByteFight.

Usage:
    uv run python src/scripts/train.py --game tictactoe --epochs 100
    uv run python src/scripts/train.py --game connect4 --epochs 100
    uv run python src/scripts/train.py --game bytefight --epochs 100
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from siebren import (
    ByteFightSelfPlay,
    Connect4SelfPlay,
    PyReplayBuffer,
    TicTacToeSelfPlay,
    sample_bytefight,
    sample_connect4,
    sample_tictactoe,
)


@dataclass
class TrainConfig:
    game: str = "tictactoe"
    epochs: int = 100
    samples_per_epoch: int = 4096
    train_batch_size: int = 256
    train_steps_per_epoch: int = 16
    replay_buffer_capacity: int = 100_000
    lr: float = 1e-3
    num_threads: int = 32
    workers_per_thread: int = 16
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_project: str = "siebren"
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 10
    resume: Optional[str] = None


class TicTacToeNet(nn.Module):
    """Simple MLP for TicTacToe (9 input cells -> 9 policy logits + 1 value)."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, 9)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 9) int8 board state

        Returns:
            policy: (B, 9) action logits (not softmaxed)
            value: (B,) position evaluation in [-1, 1]
        """
        x = x.float()
        h = self.trunk(x)
        policy = self.policy_head(h)
        value = self.value_head(h).squeeze(-1).tanh()
        return policy, value


class Connect4Net(nn.Module):
    """Simple CNN for Connect4 (6x7 board -> 7 policy logits + 1 value)."""

    def __init__(self, channels: int = 64):
        super().__init__()
        # Input: (B, 1, 6, 7)
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.policy_head = nn.Linear(channels, 7)
        self.value_head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 6, 7) int8 board state

        Returns:
            policy: (B, 7) action logits (not softmaxed)
            value: (B,) position evaluation in [-1, 1]
        """
        x = x.float().unsqueeze(1)  # (B, 1, 6, 7)
        h = self.conv(x)  # (B, C, 6, 7)
        h = self.pool(h).squeeze(-1).squeeze(-1)  # (B, C)
        policy = self.policy_head(h)
        value = self.value_head(h).squeeze(-1).tanh()
        return policy, value


class ByteFightNet(nn.Module):
    """MLP for ByteFight (18 heuristic inputs -> 11 policy logits + 1 value)."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(18, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, 11)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 18) float32 heuristic features

        Returns:
            policy: (B, 11) action logits (not softmaxed)
            value: (B,) position evaluation in [-1, 1]
        """
        h = self.trunk(x)
        policy = self.policy_head(h)
        value = self.value_head(h).squeeze(-1).tanh()
        return policy, value


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainConfig,
    replay_buffer: "PyReplayBuffer",
    checkpoint_dir: Path,
) -> None:
    """Save model and training state."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(config),
    }

    # Save checkpoint
    path = checkpoint_dir / f"checkpoint_epoch{epoch:04d}.pt"
    torch.save(checkpoint, path)

    # Also save as latest
    latest = checkpoint_dir / "latest.pt"
    torch.save(checkpoint, latest)

    # Save replay buffer
    buffer_path = checkpoint_dir / f"replay_buffer_epoch{epoch:04d}.bin"
    replay_buffer.save(str(buffer_path))

    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    replay_buffer: Optional["PyReplayBuffer"] = None,
    buffer_path: Optional[str] = None,
) -> int:
    """Load model checkpoint. Returns epoch number."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if replay_buffer is not None and buffer_path is not None:
        loaded = replay_buffer.load(buffer_path)
        print(f"Loaded {loaded} samples from replay buffer")

    return checkpoint["epoch"]


def make_execute_model(model: nn.Module, device: str, num_actions: int):
    """Create the execute_model callback for self-play.

    The callback is called from Rust with batched observations.
    It runs inference and returns (policy, value) numpy arrays.
    """

    @torch.no_grad()
    def execute_model(
        obs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # obs: (256, 9) for TicTacToe, (256, 6, 7) for Connect4, (256, 18) for ByteFight
        x = torch.from_numpy(obs).to(device)
        policy_logits, value = model(x)

        # Softmax policy and convert to numpy
        policy = F.softmax(policy_logits, dim=-1).cpu().numpy()
        value = value.cpu().numpy()

        return policy, value

    return execute_model


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay_buffer: PyReplayBuffer,
    sample_fn: Callable,
    batch_size: int,
    device: str,
    step: int,
) -> Dict[str, float]:
    """One training step. Returns dict of losses."""
    model.train()

    # Sample from replay buffer (converts notations to observations in Rust)
    obs, policies, values = sample_fn(replay_buffer, batch_size, step)

    # Move to device
    obs = torch.from_numpy(obs).to(device)
    target_policies = torch.from_numpy(policies).to(device)
    target_values = torch.from_numpy(values).to(device)

    # Forward pass
    pred_logits, pred_values = model(obs)

    # Policy loss: cross-entropy with target distribution
    # log_softmax for numerical stability
    log_probs = F.log_softmax(pred_logits, dim=-1)
    policy_loss = -(target_policies * log_probs).sum(dim=-1).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(pred_values, target_values)

    # Total loss
    total_loss = policy_loss + value_loss

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "total_loss": total_loss.item(),
    }


def train(config: TrainConfig):
    """Main training loop."""
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        config=vars(config),
        name=f"{config.game}-{time.strftime('%Y%m%d-%H%M%S')}",
    )

    # Create replay buffer
    replay_buffer = PyReplayBuffer(config.replay_buffer_capacity)

    # Setup model, selfplay, and sampling function based on game
    if config.game == "tictactoe":
        model = TicTacToeNet().to(config.device)
        selfplay = TicTacToeSelfPlay(
            num_threads=config.num_threads,
            workers_per_thread=config.workers_per_thread,
            seed=config.seed,
        )
        num_actions = 9
        sample_fn = sample_tictactoe
    elif config.game == "connect4":
        model = Connect4Net().to(config.device)
        selfplay = Connect4SelfPlay(
            num_threads=config.num_threads,
            workers_per_thread=config.workers_per_thread,
            seed=config.seed,
        )
        num_actions = 7
        sample_fn = sample_connect4
    elif config.game == "bytefight":
        model = ByteFightNet().to(config.device)
        selfplay = ByteFightSelfPlay(
            num_threads=config.num_threads,
            workers_per_thread=config.workers_per_thread,
            seed=config.seed,
        )
        num_actions = 11
        sample_fn = sample_bytefight
    else:
        raise ValueError(f"Unknown game: {config.game}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    execute_model = make_execute_model(model, config.device, num_actions)

    # Setup checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)

    # Resume from checkpoint if specified
    start_epoch = 0
    if config.resume:
        checkpoint_path = Path(config.resume)
        buffer_path = checkpoint_path.parent / checkpoint_path.name.replace(
            "checkpoint_", "replay_buffer_"
        ).replace(".pt", ".bin")
        start_epoch = load_checkpoint(
            model,
            optimizer,
            str(checkpoint_path),
            replay_buffer,
            str(buffer_path) if buffer_path.exists() else None,
        )
        start_epoch += 1  # Start from the next epoch
        print(f"Resumed from epoch {start_epoch - 1}, starting at epoch {start_epoch}")

    # Track metrics
    total_games = 0
    total_samples = 0
    total_batches = 0
    global_step = start_epoch

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()

        # Self-play phase
        model.eval()
        batch_count = [0]

        def counting_execute_model(obs):
            batch_count[0] += 1
            return execute_model(obs)

        selfplay_start = time.time()
        games_completed, samples_collected = selfplay.play_games(
            replay_buffer=replay_buffer,
            num_samples=config.samples_per_epoch,
            execute_model=counting_execute_model,
        )
        selfplay_time = time.time() - selfplay_start

        total_games += games_completed
        total_samples += samples_collected
        total_batches += batch_count[0]

        # Log self-play metrics
        selfplay_metrics = {
            "epoch": epoch,
            "selfplay/games_completed": games_completed,
            "selfplay/samples_collected": samples_collected,
            "selfplay/batches": batch_count[0],
            "selfplay/time_sec": selfplay_time,
            "selfplay/games_per_sec": games_completed / selfplay_time,
            "selfplay/samples_per_sec": samples_collected / selfplay_time,
            "selfplay/batches_per_sec": batch_count[0] / selfplay_time,
            "total/games": total_games,
            "total/samples": total_samples,
            "total/batches": total_batches,
            "replay_buffer/size": len(replay_buffer),
        }
        wandb.log(selfplay_metrics, step=global_step)

        # Training phase
        if len(replay_buffer) >= config.train_batch_size:
            model.train()
            train_start = time.time()

            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_total_loss = 0.0

            for step in range(config.train_steps_per_epoch):
                losses = train_step(
                    model,
                    optimizer,
                    replay_buffer,
                    sample_fn,
                    config.train_batch_size,
                    config.device,
                    global_step + step,
                )
                epoch_policy_loss += losses["policy_loss"]
                epoch_value_loss += losses["value_loss"]
                epoch_total_loss += losses["total_loss"]

            train_time = time.time() - train_start
            num_steps = config.train_steps_per_epoch

            # Log training metrics
            train_metrics = {
                "train/policy_loss": epoch_policy_loss / num_steps,
                "train/value_loss": epoch_value_loss / num_steps,
                "train/total_loss": epoch_total_loss / num_steps,
                "train/time_sec": train_time,
                "train/steps_per_sec": num_steps / train_time,
            }
            wandb.log(train_metrics, step=global_step)

        global_step += 1
        epoch_time = time.time() - epoch_start

        # Print progress
        buffer_pct = 100 * len(replay_buffer) / config.replay_buffer_capacity
        print(
            f"Epoch {epoch}: {games_completed} games, {samples_collected} samples, "
            f"buffer {len(replay_buffer)}/{config.replay_buffer_capacity} ({buffer_pct:.1f}%), "
            f"{epoch_time:.1f}s"
        )

        # Save checkpoint periodically
        if (epoch + 1) % config.checkpoint_every == 0:
            save_checkpoint(
                model, optimizer, epoch, config, replay_buffer, checkpoint_dir
            )

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero-style agent")
    parser.add_argument(
        "--game",
        type=str,
        default="tictactoe",
        choices=["tictactoe", "connect4", "bytefight"],
        help="Game to train on",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=4096,
        help="Target samples to collect per epoch",
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument(
        "--train-steps-per-epoch",
        type=int,
        default=16,
        help="Training steps per epoch",
    )
    parser.add_argument(
        "--replay-buffer-capacity",
        type=int,
        default=100_000,
        help="Replay buffer capacity",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--num-threads", type=int, default=32, help="Number of worker threads"
    )
    parser.add_argument(
        "--workers-per-thread", type=int, default=16, help="Workers per thread"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="siebren", help="W&B project name"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    config = TrainConfig(
        game=args.game,
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        train_batch_size=args.train_batch_size,
        train_steps_per_epoch=args.train_steps_per_epoch,
        replay_buffer_capacity=args.replay_buffer_capacity,
        lr=args.lr,
        num_threads=args.num_threads,
        workers_per_thread=args.workers_per_thread,
        seed=args.seed,
        device=args.device,
        wandb_project=args.wandb_project,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )

    train(config)


if __name__ == "__main__":
    main()

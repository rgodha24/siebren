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
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
from siebren import (
    ByteFightReplayBuffer,
    ByteFightSelfPlay,
    Connect4ReplayBuffer,
    Connect4SelfPlay,
    TicTacToeReplayBuffer,
    TicTacToeSelfPlay,
)

# Type alias for the typed replay buffers
ReplayBuffer = Union[TicTacToeReplayBuffer, Connect4ReplayBuffer, ByteFightReplayBuffer]


@dataclass
class TrainConfig:
    game: str = "tictactoe"
    epochs: int = 100
    samples_per_epoch: int = 5_000_000
    train_batch_size: int = 2048
    train_steps_per_epoch: int = 256
    replay_buffer_capacity: int = 50_000_000
    lr: float = 1e-3
    value_loss_weight: float = 1.0
    l2_weight: float = 1e-4
    num_threads: int = 32
    workers_per_thread: int = 256
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_project: str = "siebren"
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 10
    resume: Optional[str] = None
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_fullgraph: bool = False
    cudagraphs: bool = False
    matmul_precision: str = "high"


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
    replay_buffer: ReplayBuffer,
    checkpoint_dir: Path,
) -> None:
    # this shit uses SO MUCH STORAGE we dont need it
    return
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

    # Save replay buffer (only unsaved samples)
    buffer_path = checkpoint_dir / f"replay_buffer_epoch{epoch:04d}.bin"
    samples_saved = replay_buffer.save(str(buffer_path), epoch)
    replay_buffer.mark_saved()

    print(f"Saved checkpoint to {path} ({samples_saved} new samples)")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    replay_buffer: Optional[ReplayBuffer] = None,
    buffer_path: Optional[str] = None,
) -> int:
    """Load model checkpoint. Returns epoch number."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if replay_buffer is not None and buffer_path is not None:
        samples_loaded, generation_id = replay_buffer.load(buffer_path)
        print(
            f"Loaded {samples_loaded} samples from replay buffer (generation {generation_id})"
        )

    return checkpoint["epoch"]


def make_execute_model(model: nn.Module, device: str, num_actions: int):
    """Create the execute_model callback for self-play.

    The callback is called from Rust with batched observations.
    It runs inference and returns (policy, value) numpy arrays.
    """

    @torch.inference_mode()
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


def configure_torch(config: TrainConfig) -> None:
    if config.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision(config.matmul_precision)
        except Exception as exc:
            print(
                f"Warning: could not set matmul precision to {config.matmul_precision}: {exc}"
            )


def maybe_compile_model(model: nn.Module, config: TrainConfig) -> nn.Module:
    if not config.compile:
        return model
    if not hasattr(torch, "compile"):
        print("torch.compile unavailable; running eager.")
        return model
    if config.device.startswith("cuda"):
        try:
            import torch._inductor.config as inductor_config

            inductor_config.triton.cudagraphs = bool(config.cudagraphs)
            if hasattr(inductor_config.triton, "cudagraph_trees"):
                inductor_config.triton.cudagraph_trees = bool(config.cudagraphs)
        except Exception as exc:
            print(f"Warning: unable to configure inductor cudagraphs: {exc}")
    try:
        compiled = torch.compile(
            model, mode=config.compile_mode, fullgraph=config.compile_fullgraph
        )
        return compiled
    except Exception as exc:
        print(f"torch.compile failed ({exc}); running eager.")
        return model


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    device: str,
    step: int,
    value_loss_weight: float,
    l2_weight: float,
) -> Dict[str, float]:
    """One training step. Returns dict of losses."""
    model.train()

    # Sample from replay buffer (converts notations to observations in Rust)
    obs, policies, values = replay_buffer.sample(batch_size, step)

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

    # L2 regularization
    l2_loss = torch.zeros((), device=device)
    if l2_weight > 0.0:
        for param in model.parameters():
            if param.requires_grad:
                l2_loss = l2_loss + param.pow(2).sum()

    # Total loss
    total_loss = policy_loss + value_loss_weight * value_loss + l2_weight * l2_loss

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "l2_loss": l2_loss.item(),
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

    configure_torch(config)

    # Setup model, selfplay, and replay buffer based on game
    if config.game == "tictactoe":
        model = TicTacToeNet().to(config.device)
        selfplay = TicTacToeSelfPlay(
            num_threads=config.num_threads,
            workers_per_thread=config.workers_per_thread,
            seed=config.seed,
        )
        num_actions = 9
        replay_buffer: ReplayBuffer = TicTacToeReplayBuffer(
            config.replay_buffer_capacity
        )
    elif config.game == "connect4":
        model = Connect4Net().to(config.device)
        selfplay = Connect4SelfPlay(
            num_threads=config.num_threads,
            workers_per_thread=config.workers_per_thread,
            seed=config.seed,
        )
        num_actions = 7
        replay_buffer = Connect4ReplayBuffer(config.replay_buffer_capacity)
    elif config.game == "bytefight":
        model = ByteFightNet().to(config.device)
        selfplay = ByteFightSelfPlay(
            num_threads=config.num_threads,
            workers_per_thread=config.workers_per_thread,
            seed=config.seed,
        )
        num_actions = 11
        replay_buffer = ByteFightReplayBuffer(config.replay_buffer_capacity)
    else:
        raise ValueError(f"Unknown game: {config.game}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

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

    if config.cudagraphs and not config.compile:
        print("Warning: --cudagraphs set without --compile; cudagraphs disabled.")
        config.cudagraphs = False
    if config.cudagraphs and (config.num_threads * config.workers_per_thread) > 1:
        print(
            "Warning: cudagraphs with multithreaded self-play is unstable; "
            "disabling cudagraphs."
        )
        config.cudagraphs = False

    model = maybe_compile_model(model, config)
    execute_model = make_execute_model(model, config.device, num_actions)

    # Track metrics
    total_games = 0
    total_batches = 0
    global_step = start_epoch

    # Batch counter - reused across epochs to avoid creating new closures
    batch_count = [0]

    def counting_execute_model(obs):
        batch_count[0] += 1
        return execute_model(obs)

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()

        # Self-play phase
        model.eval()
        batch_count[0] = 0  # Reset counter for this epoch

        selfplay_start = time.time()
        selfplay_result = selfplay.play_games(
            replay_buffer=replay_buffer,
            num_samples=config.samples_per_epoch,
            execute_model=counting_execute_model,
        )
        if len(selfplay_result) == 2:
            games_completed, samples_collected = selfplay_result
            executor_stats = {
                "poll_rounds": 0,
                "futures_polled": 0,
                "poll_ready": 0,
                "poll_pending": 0,
                "wait_count": 0,
            }
        else:
            games_completed, samples_collected, executor_stats = selfplay_result
        selfplay_time = time.time() - selfplay_start

        total_games += games_completed
        total_batches += batch_count[0]

        moves_per_game = (
            samples_collected / games_completed if games_completed > 0 else 0.0
        )

        # Log self-play metrics
        selfplay_metrics = {
            "epoch": epoch,
            "selfplay/games_completed": games_completed,
            "selfplay/samples_collected": samples_collected,
            "selfplay/moves_per_game": moves_per_game,
            "selfplay/batches": batch_count[0],
            "selfplay/time_sec": selfplay_time,
            "selfplay/games_per_sec": games_completed / selfplay_time,
            "selfplay/samples_per_sec": samples_collected / selfplay_time,
            "selfplay/batches_per_sec": batch_count[0] / selfplay_time,
            "selfplay/executor_poll_rounds": executor_stats["poll_rounds"],
            "selfplay/executor_futures_polled": executor_stats["futures_polled"],
            "selfplay/executor_poll_ready": executor_stats["poll_ready"],
            "selfplay/executor_poll_pending": executor_stats["poll_pending"],
            "selfplay/executor_wait_count": executor_stats["wait_count"],
            "total/games": total_games,
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
            epoch_l2_loss = 0.0
            epoch_total_loss = 0.0

            for step in range(config.train_steps_per_epoch):
                losses = train_step(
                    model,
                    optimizer,
                    replay_buffer,
                    config.train_batch_size,
                    config.device,
                    global_step + step,
                    config.value_loss_weight,
                    config.l2_weight,
                )
                epoch_policy_loss += losses["policy_loss"]
                epoch_value_loss += losses["value_loss"]
                epoch_l2_loss += losses["l2_loss"]
                epoch_total_loss += losses["total_loss"]

            train_time = time.time() - train_start
            num_steps = config.train_steps_per_epoch

            # Log training metrics
            train_metrics = {
                "train/policy_loss": epoch_policy_loss / num_steps,
                "train/value_loss": epoch_value_loss / num_steps,
                "train/l2_loss": epoch_l2_loss / num_steps,
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
            f"{moves_per_game:.1f} moves/game, "
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
        default=5_000_000,
        help="Target samples to collect per epoch",
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=2048, help="Training batch size"
    )
    parser.add_argument(
        "--train-steps-per-epoch",
        type=int,
        default=256,
        help="Training steps per epoch",
    )
    parser.add_argument(
        "--replay-buffer-capacity",
        type=int,
        default=50_000_000,
        help="Replay buffer capacity",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=1.0,
        help="Weight for value loss term",
    )
    parser.add_argument(
        "--l2-weight",
        type=float,
        default=1e-4,
        help="L2 regularization weight",
    )
    parser.add_argument(
        "--num-threads", type=int, default=32, help="Number of worker threads"
    )
    parser.add_argument(
        "--workers-per-thread", type=int, default=256, help="Workers per thread"
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
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the model",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help="torch.compile mode",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help="Require full graph capture in torch.compile",
    )
    parser.add_argument(
        "--cudagraphs",
        action="store_true",
        help="Enable inductor cudagraphs (requires --compile)",
    )
    parser.add_argument(
        "--matmul-precision",
        type=str,
        default="high",
        choices=["highest", "high", "medium"],
        help="Set float32 matmul precision (CUDA only)",
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
        value_loss_weight=args.value_loss_weight,
        l2_weight=args.l2_weight,
        num_threads=args.num_threads,
        workers_per_thread=args.workers_per_thread,
        seed=args.seed,
        device=args.device,
        wandb_project=args.wandb_project,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        compile=args.compile,
        compile_mode=args.compile_mode,
        compile_fullgraph=args.compile_fullgraph,
        cudagraphs=args.cudagraphs,
        matmul_precision=args.matmul_precision,
    )

    train(config)


if __name__ == "__main__":
    main()

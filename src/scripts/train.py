"""AlphaZero-style training script for TicTacToe and Connect4.

Usage:
    uv run python src/scripts/train.py --game tictactoe --epochs 100
    uv run python src/scripts/train.py --game connect4 --epochs 100
"""

import argparse
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from siebren import Connect4SelfPlay, TicTacToeSelfPlay


@dataclass
class TrainConfig:
    game: str = "tictactoe"
    epochs: int = 100
    games_per_epoch: int = 256
    train_batch_size: int = 256
    train_steps_per_epoch: int = 10
    lr: float = 1e-3
    num_threads: int = 32
    workers_per_thread: int = 16
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_project: str = "siebren"


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


def make_execute_model(model: nn.Module, device: str, num_actions: int):
    """Create the execute_model callback for self-play.

    The callback is called from Rust with batched observations.
    It runs inference and returns (policy, value) numpy arrays.
    """

    @torch.no_grad()
    def execute_model(
        obs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # obs: (256, 9) for TicTacToe, (256, 6, 7) for Connect4
        x = torch.from_numpy(obs).to(device)
        policy_logits, value = model(x)

        # Softmax policy and convert to numpy
        policy = F.softmax(policy_logits, dim=-1).cpu().numpy()
        value = value.cpu().numpy()

        return policy, value

    return execute_model


def train(config: TrainConfig):
    """Main training loop."""
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        config=vars(config),
        name=f"{config.game}-{time.strftime('%Y%m%d-%H%M%S')}",
    )

    # Setup model and optimizer
    if config.game == "tictactoe":
        model = TicTacToeNet().to(config.device)
        selfplay = TicTacToeSelfPlay(
            num_threads=config.num_threads,
            workers_per_thread=config.workers_per_thread,
            seed=config.seed,
        )
        num_actions = 9
    elif config.game == "connect4":
        model = Connect4Net().to(config.device)
        selfplay = Connect4SelfPlay(
            num_threads=config.num_threads,
            workers_per_thread=config.workers_per_thread,
            seed=config.seed,
        )
        num_actions = 7
    else:
        raise ValueError(f"Unknown game: {config.game}")

    optimizer = torch.optim.Muon(model.parameters(), lr=config.lr)
    execute_model = make_execute_model(model, config.device, num_actions)

    # Track metrics
    total_games = 0
    total_batches = 0

    for epoch in range(config.epochs):
        epoch_start = time.time()

        # Self-play phase
        model.eval()
        batch_count = [0]

        def counting_execute_model(obs):
            batch_count[0] += 1
            return execute_model(obs)

        selfplay_start = time.time()
        games_completed = selfplay.play_games(
            num_games=config.games_per_epoch,
            execute_model=counting_execute_model,
        )
        selfplay_time = time.time() - selfplay_start

        total_games += games_completed
        total_batches += batch_count[0]

        # Log self-play metrics
        wandb.log(
            {
                "epoch": epoch,
                "selfplay/games_completed": games_completed,
                "selfplay/batches": batch_count[0],
                "selfplay/time_sec": selfplay_time,
                "selfplay/games_per_sec": games_completed / selfplay_time,
                "selfplay/batches_per_sec": batch_count[0] / selfplay_time,
                "total/games": total_games,
                "total/batches": total_batches,
            }
        )

        # Training phase (placeholder - we don't have samples returned yet)
        # TODO: Once samples are returned from Rust, add training loop here
        # model.train()
        # for step in range(config.train_steps_per_epoch):
        #     ...

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch}: {games_completed} games, "
            f"{batch_count[0]} batches, {epoch_time:.1f}s"
        )

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero-style agent")
    parser.add_argument(
        "--game",
        type=str,
        default="tictactoe",
        choices=["tictactoe", "connect4"],
        help="Game to train on",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--games-per-epoch", type=int, default=256, help="Games per epoch"
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

    args = parser.parse_args()

    config = TrainConfig(
        game=args.game,
        epochs=args.epochs,
        games_per_epoch=args.games_per_epoch,
        lr=args.lr,
        num_threads=args.num_threads,
        workers_per_thread=args.workers_per_thread,
        seed=args.seed,
        device=args.device,
        wandb_project=args.wandb_project,
    )

    train(config)


if __name__ == "__main__":
    main()

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EPOCHS = 1000
ACTIONS = 10
ENVS = 1024
BATCH_SIZE = 128_000
GAMES_PER_STEP = 4096
TRAIN_STEPS = 10


class ReplayBuffer:
    def sample(
        self, amt: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]: ...
    def __len__(self):
        return 0


def SelfPlay_snake(envs: int) -> "SelfPlay":
    return SelfPlay(envs)


class SelfPlay:
    envs: int
    replay_buffer: ReplayBuffer

    def __init__(self, envs: int) -> None:
        self.envs = envs

    def play_games(
        self, replay_buffer: ReplayBuffer, num_games: int, model_name: str
    ): ...


class SnakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.board = nn.Sequential(
            nn.Conv2d(7, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )  # (B, 64)
        self.history = nn.LSTM(18, 32, batch_first=True)  # (B, 32)
        self.heuristic = nn.Sequential(
            nn.Linear(18, 32), nn.ReLU(), nn.Linear(32, 32)
        )  # (B, 32)

        self.trunk = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
        self.policy_head = nn.Sequential(nn.Linear(32, ACTIONS), nn.LogSoftmax(dim=1))
        self.value_head = nn.Sequential(nn.Linear(32, 1), nn.Tanh())

    def forward(
        self,
        board: torch.Tensor,  # (B, 7, 32, 32)
        heuristic_history: torch.Tensor,  # (B, seq_len, 18)
    ):
        board_emb = self.board(board)  # (B, 64)
        _, (history_emb, _) = self.history(heuristic_history)
        history_emb = history_emb.squeeze(0)

        assert heuristic_history.size(1) >= 1, (
            "heuristic_history must have at least 1 timestep"
        )
        last_heuristic = heuristic_history[:, -1, :]  # (B, 18)
        heuristic_emb = self.heuristic(last_heuristic)  # (B, 32)
        trunk_emb = torch.cat(
            [board_emb, history_emb, heuristic_emb], dim=1
        )  # (B, 128)
        trunk_emb = self.trunk(trunk_emb)  # (B, 32)
        policy = self.policy_head(trunk_emb)  # (B, 10)
        value = self.value_head(trunk_emb)  # (B, 1)
        return policy, value


def save_model(model: nn.Module, epoch: int) -> str: ...


model = SnakeModel().to(device="cuda", dtype=torch.bfloat16)
replay_buffer = ReplayBuffer()
self_play = SelfPlay_snake(ENVS)
optimizer = torch.optim.Muon(model.parameters())

for epoch in range(EPOCHS):
    model_name = save_model(model, epoch)
    self_play.play_games(replay_buffer, GAMES_PER_STEP, model_name)
    if len(replay_buffer) <= BATCH_SIZE:
        continue

    for _ in range(TRAIN_STEPS):
        states, target_policies, target_values = replay_buffer.sample(
            BATCH_SIZE // TRAIN_STEPS
        )
        states = {k: v.to("cuda") for (k, v) in states.items()}
        target_policies = target_policies.to("cuda")
        target_values = target_values.to("cuda")

        log_policies, values = model.forward(**states)
        policy_loss = -torch.mean(torch.sum(target_policies * log_policies, dim=1))
        value_loss = F.mse_loss(values, target_values)

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


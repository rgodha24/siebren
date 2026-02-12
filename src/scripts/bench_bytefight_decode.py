#!/usr/bin/env python

import argparse
import time
from typing import cast

import torch
import torch.nn as nn


class ByteFightNet(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.register_buffer(
            "_bit_shifts",
            torch.arange(8, dtype=torch.uint8).view(1, 8, 1, 1),
            persistent=False,
        )
        input_dim = 8 * 16 * 16 + 8 + 18
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, 7)
        self.value_head = nn.Linear(hidden_dim, 1)

    def decode_observation(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 18 or x.shape[2] != 16:
            raise ValueError(
                f"Expected ByteFight obs shape (B, 18, 16), got {tuple(x.shape)}"
            )
        if x.dtype != torch.uint8:
            x = x.to(torch.uint8)

        board = x[:, :16, :]
        meta = x[:, 16:, :].reshape(x.shape[0], 32)
        direction = meta[:, :8].to(torch.float32)
        heuristics = (meta[:, 8:26].to(torch.float32) - 128.0) / 127.0

        shifts = cast(torch.Tensor, self._bit_shifts)
        bitplanes = torch.bitwise_and(
            torch.bitwise_right_shift(board.unsqueeze(1), shifts),
            1,
        ).to(torch.float32)

        return torch.cat((bitplanes.flatten(1), direction, heuristics), dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.decode_observation(x))
        policy = self.policy_head(h)
        value = self.value_head(h).squeeze(-1).tanh()
        return policy, value


def sync_if_cuda(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.inference_mode()
def run_benchmark(
    device: str,
    batch_size: int,
    iterations: int,
    warmup: int,
    hidden_dim: int,
) -> None:
    model = ByteFightNet(hidden_dim=hidden_dim).to(device)
    model.eval()
    x = torch.randint(0, 256, (batch_size, 18, 16), dtype=torch.uint8, device=device)

    for _ in range(warmup):
        _ = model.decode_observation(x)
        _ = model(x)
    sync_if_cuda(device)

    start = time.perf_counter()
    for _ in range(iterations):
        _ = model.decode_observation(x)
    sync_if_cuda(device)
    decode_sec = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        _ = model(x)
    sync_if_cuda(device)
    forward_sec = time.perf_counter() - start

    total_obs = batch_size * iterations
    decode_obs_per_sec = total_obs / decode_sec
    forward_obs_per_sec = total_obs / forward_sec
    decode_gbps = decode_obs_per_sec * (18 * 16) / 1e9
    forward_gbps = forward_obs_per_sec * (18 * 16) / 1e9
    decode_share = 100.0 * decode_sec / max(forward_sec, 1e-12)

    print("ByteFight decode benchmark")
    print(
        f"  device={device} hidden_dim={hidden_dim} batch={batch_size} iters={iterations} warmup={warmup}"
    )
    print(
        f"  decode only: {decode_obs_per_sec:,.0f} obs/s ({decode_gbps:.2f} GB/s input)"
    )
    print(
        f"  full forward: {forward_obs_per_sec:,.0f} obs/s ({forward_gbps:.2f} GB/s input)"
    )
    print(f"  decode_time/full_forward_time: {decode_share:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ByteFight observation decode path"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    run_benchmark(
        device=args.device,
        batch_size=args.batch_size,
        iterations=args.iters,
        warmup=args.warmup,
        hidden_dim=args.hidden_dim,
    )


if __name__ == "__main__":
    main()

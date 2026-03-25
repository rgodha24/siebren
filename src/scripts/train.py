"""AlphaZero-style training script for TicTacToe, Connect4, and ByteFight.

Usage:
    uv run python src/scripts/train.py --game tictactoe --epochs 100
    uv run python src/scripts/train.py --game connect4 --epochs 100
    uv run python src/scripts/train.py --game bytefight --epochs 100
"""

# pyright: reportMissingImports=false, reportInvalidTypeForm=false, reportIndexIssue=false, reportAttributeAccessIssue=false

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

import wandb
from siebren import (
    EphemeralReplayBuffer,
    SelfPlay,
)


@dataclass
class TrainConfig:
    game: str = "bytefight"
    epochs: int = 300
    samples_per_epoch: int = 1_000_000
    train_batch_size: int = 2048
    train_steps_per_epoch: int = 512
    replay_buffer_capacity: int = 10_000_000
    lr: float = 1e-3
    value_loss_weight: float = 2.0
    l2_weight: float = 1e-4
    num_threads: int = 32
    workers_per_thread: int = 96
    hidden_dim: Optional[int] = None
    mcts_num_simulations: int = 64
    mcts_c_puct: float = 1.5
    mcts_dirichlet_alpha: float = 0.3
    mcts_dirichlet_epsilon: float = 0.25
    selfplay_temperature: float = 1.0
    selfplay_exploration_moves: int = 30
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_project: str = "siebren"
    run_name: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 10
    resume: Optional[str] = None
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_fullgraph: bool = False
    cudagraphs: bool = False
    selfplay_backend: str = "rust-cudagraph"
    selfplay_precision: str = "fp16"
    matmul_precision: str = "high"
    num_gpus: int = 0  # 0 means auto-detect from torch.cuda.device_count()


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
    """MLP for compact ByteFight observations (18x16 uint8)."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        input_dim = 8 * 16 * 16 + 8 + 18
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, 7)
        self.value_head = nn.Linear(hidden_dim, 1)
        self._triton_decode_lanes = 256
        self._triton_decode_num_warps = 4
        self._decode_out: Optional[torch.Tensor] = None

    def _get_decode_out(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        cached = self._decode_out
        if (
            cached is None
            or cached.device != x.device
            or cached.dtype != torch.float32
            or cached.shape[0] < batch
        ):
            cached = torch.empty((batch, 2074), dtype=torch.float32, device=x.device)
            self._decode_out = cached
        return cached[:batch]

    def decode_observation(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected ByteFight obs shape (B, 18, 16), got {tuple(x.shape)}"
            )
        if x.shape[1] != 18 or x.shape[2] != 16:
            raise ValueError(
                f"Expected ByteFight obs shape (B, 18, 16), got {tuple(x.shape)}"
            )
        if not x.is_cuda:
            raise ValueError("ByteFight decode requires CUDA tensor input")
        if x.dtype != torch.uint8:
            x = x.to(torch.uint8)

        out = self._get_decode_out(x)
        _decode_bytefight_triton(
            x,
            out,
            self._triton_decode_lanes,
            self._triton_decode_num_warps,
        )
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 18, 16) uint8 compact observations

        Returns:
            policy: (B, 7) action logits (not softmaxed)
            value: (B,) position evaluation in [-1, 1]
        """
        h = self.trunk(self.decode_observation(x))
        policy = self.policy_head(h)
        value = self.value_head(h).squeeze(-1).tanh()
        return policy, value


@triton.jit
def _bytefight_decode_kernel(
    obs_ptr,
    out_ptr,
    batch,
    lanes: tl.constexpr,
):
    row = tl.program_id(0)
    seg = tl.program_id(1)
    lane = tl.arange(0, lanes)
    mask_row = row < batch

    row_obs_base = row * 288
    row_out_base = row * 2074

    mask_board = mask_row & (seg < 8)
    obs_board_idx = row_obs_base + lane
    cell = tl.load(obs_ptr + obs_board_idx, mask=mask_board, other=0).to(tl.int32)
    board_val = ((cell >> seg) & 1).to(tl.float32)
    board_out_idx = row_out_base + seg * 256 + lane
    tl.store(out_ptr + board_out_idx, board_val, mask=mask_board)

    mask_dir = mask_row & (seg == 8) & (lane < 8)
    dir_val = tl.load(obs_ptr + row_obs_base + 256 + lane, mask=mask_dir, other=0).to(
        tl.float32
    )
    tl.store(out_ptr + row_out_base + 2048 + lane, dir_val, mask=mask_dir)

    mask_heur = mask_row & (seg == 8) & (lane < 18)
    heur_q = tl.load(obs_ptr + row_obs_base + 264 + lane, mask=mask_heur, other=128).to(
        tl.float32
    )
    heur_val = (heur_q - 128.0) / 127.0
    tl.store(out_ptr + row_out_base + 2056 + lane, heur_val, mask=mask_heur)


def _decode_bytefight_triton(
    obs: torch.Tensor,
    out: torch.Tensor,
    lanes: int,
    num_warps: int,
) -> None:
    batch = obs.shape[0]
    grid = (batch, 9)
    _bytefight_decode_kernel[grid](
        obs,
        out,
        batch,
        lanes=lanes,
        num_warps=num_warps,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainConfig,
    _replay_buffer: EphemeralReplayBuffer,
    checkpoint_dir: Path,
) -> None:
    """Save model and optimizer state."""
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

    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    replay_buffer: Optional[EphemeralReplayBuffer] = None,
    buffer_path: Optional[str] = None,
) -> int:
    """Load model checkpoint. Returns epoch number."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if replay_buffer is not None and buffer_path is not None:
        samples_loaded, generation_id = replay_buffer.load(buffer_path)  # type: ignore[attr-defined]
        print(
            f"Loaded {samples_loaded} samples from replay buffer (generation {generation_id})"
        )

    return checkpoint["epoch"]


def make_execute_model(
    model: nn.Module,
    device: str,
    num_actions: int,
    use_inference_cuda_graph: bool,
    selfplay_precision: str,
):
    """Create the execute_model callback for self-play.

    The callback is called from Rust with batched observations.
    It runs inference and returns (policy, value) numpy arrays.
    """

    autocast_dtype: Optional[torch.dtype] = None
    if device.startswith("cuda"):
        if selfplay_precision == "fp16":
            autocast_dtype = torch.float16
        elif selfplay_precision == "bf16":
            if torch.cuda.is_bf16_supported():
                autocast_dtype = torch.bfloat16
            else:
                print("Warning: bf16 autocast unsupported; falling back to fp16.")
                autocast_dtype = torch.float16

    def model_forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if autocast_dtype is None:
            return model(x)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            return model(x)

    @torch.inference_mode()
    def execute_model(
        obs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # obs: (256, 9) for TicTacToe, (256, 6, 7) for Connect4, (256, 18, 16) for ByteFight
        x = torch.from_numpy(obs).to(device)
        policy_logits, value = model_forward(x)

        # Softmax policy and convert to numpy
        policy = F.softmax(policy_logits, dim=-1).cpu().numpy()
        value = value.cpu().numpy()

        return policy, value

    if not (use_inference_cuda_graph and device.startswith("cuda")):
        return execute_model

    callback_lock = threading.Lock()

    graph_state: Dict[str, object] = {
        "ready": False,
        "shape": None,
        "fallback": False,
    }

    @torch.inference_mode()
    def execute_model_cuda_graph(
        obs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with callback_lock:
            if graph_state["fallback"]:
                return execute_model(obs)

            try:
                obs_shape = obs.shape
                if (not graph_state["ready"]) or graph_state["shape"] != obs_shape:
                    src = torch.from_numpy(obs)
                    device_obs = torch.empty_like(src, device=device)
                    device_policy = torch.empty(
                        (obs_shape[0], num_actions), dtype=torch.float32, device=device
                    )
                    device_value = torch.empty(
                        (obs_shape[0],), dtype=torch.float32, device=device
                    )

                    host_policy = torch.empty(
                        (obs_shape[0], num_actions),
                        dtype=torch.float32,
                        pin_memory=True,
                    )
                    host_value = torch.empty(
                        (obs_shape[0],), dtype=torch.float32, pin_memory=True
                    )
                    host_obs = torch.empty_like(src, pin_memory=True)

                    warmup_stream = torch.cuda.Stream()
                    warmup_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(warmup_stream):
                        for _ in range(5):
                            logits, value = model_forward(device_obs)
                            device_policy.copy_(F.softmax(logits, dim=-1))
                            device_value.copy_(value)
                    torch.cuda.current_stream().wait_stream(warmup_stream)

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        logits, value = model_forward(device_obs)
                        device_policy.copy_(F.softmax(logits, dim=-1))
                        device_value.copy_(value)

                    graph_state.update(
                        {
                            "ready": True,
                            "shape": obs_shape,
                            "device_obs": device_obs,
                            "device_policy": device_policy,
                            "device_value": device_value,
                            "host_policy": host_policy,
                            "host_value": host_value,
                            "host_obs": host_obs,
                            "policy_np": host_policy.numpy(),
                            "value_np": host_value.numpy(),
                            "graph": graph,
                        }
                    )

                device_obs = graph_state["device_obs"]
                device_policy = graph_state["device_policy"]
                device_value = graph_state["device_value"]
                host_policy = graph_state["host_policy"]
                host_value = graph_state["host_value"]
                host_obs = graph_state["host_obs"]
                policy_np = graph_state["policy_np"]
                value_np = graph_state["value_np"]
                graph = graph_state["graph"]

                assert isinstance(device_obs, torch.Tensor)
                assert isinstance(device_policy, torch.Tensor)
                assert isinstance(device_value, torch.Tensor)
                assert isinstance(host_policy, torch.Tensor)
                assert isinstance(host_value, torch.Tensor)
                assert isinstance(host_obs, torch.Tensor)
                assert isinstance(policy_np, np.ndarray)
                assert isinstance(value_np, np.ndarray)
                assert isinstance(graph, torch.cuda.CUDAGraph)

                host_obs.copy_(torch.from_numpy(obs), non_blocking=False)
                device_obs.copy_(host_obs, non_blocking=True)
                graph.replay()
                host_policy.copy_(device_policy, non_blocking=True)
                host_value.copy_(device_value, non_blocking=True)
                torch.cuda.current_stream().synchronize()

                return policy_np, value_np
            except Exception as exc:
                print(
                    f"Warning: inference CUDA graph failed ({exc}); "
                    "falling back to eager callback."
                )
                graph_state["fallback"] = True
                return execute_model(obs)

    return execute_model_cuda_graph


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

            triton_config = getattr(inductor_config, "triton", None)
            if triton_config is not None:
                triton_config.cudagraphs = bool(config.cudagraphs)
                if hasattr(triton_config, "cudagraph_trees"):
                    triton_config.cudagraph_trees = bool(config.cudagraphs)
        except Exception as exc:
            print(f"Warning: unable to configure inductor cudagraphs: {exc}")
    try:
        compiled = torch.compile(
            model, mode=config.compile_mode, fullgraph=config.compile_fullgraph
        )
        return cast(nn.Module, compiled)
    except Exception as exc:
        print(f"torch.compile failed ({exc}); running eager.")
        return model


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay_buffer: EphemeralReplayBuffer,
    batch_size: int,
    device: str,
    step: int,
    value_loss_weight: float,
    l2_weight: float,
) -> Dict[str, float]:
    """One training step. Returns dict of losses."""
    model.train()

    # Sample from replay buffer
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

    with torch.no_grad():
        value_pred_mean = pred_values.mean()
        value_pred_std = pred_values.std(unbiased=False)
        target_value_mean = target_values.mean()
        target_value_std = target_values.std(unbiased=False)
        value_sign_acc = ((pred_values >= 0.0) == (target_values >= 0.0)).float().mean()

        pred_centered = pred_values - value_pred_mean
        target_centered = target_values - target_value_mean
        corr_denom = (
            pred_centered.pow(2).mean().sqrt() * target_centered.pow(2).mean().sqrt()
        )
        if corr_denom.item() > 1e-8:
            value_corr = (pred_centered * target_centered).mean() / corr_denom
        else:
            value_corr = torch.zeros((), device=device)

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
        "value_pred_mean": value_pred_mean.item(),
        "value_pred_std": value_pred_std.item(),
        "target_value_mean": target_value_mean.item(),
        "target_value_std": target_value_std.item(),
        "value_sign_acc": value_sign_acc.item(),
        "value_corr": value_corr.item(),
    }


def train(config: TrainConfig):
    """Main training loop."""
    default_run_name = f"{config.game}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_name = config.run_name or default_run_name

    # Initialize wandb
    run = wandb.init(
        project=config.wandb_project,
        config=vars(config),
        name=run_name,
    )

    run_name = run.name if run is not None and run.name else default_run_name
    checkpoint_run_name = "".join(
        ch if (ch.isalnum() or ch in "-_.") else "_" for ch in run_name
    )
    checkpoint_run_name = checkpoint_run_name.strip("._") or "run"

    configure_torch(config)

    # Setup model and action space based on game
    if config.game == "tictactoe":
        hidden_dim = config.hidden_dim if config.hidden_dim is not None else 128
        model = TicTacToeNet(hidden_dim=hidden_dim).to(config.device)
        num_actions = 9
    elif config.game == "connect4":
        channels = config.hidden_dim if config.hidden_dim is not None else 64
        model = Connect4Net(channels=channels).to(config.device)
        num_actions = 7
    elif config.game == "bytefight":
        hidden_dim = config.hidden_dim if config.hidden_dim is not None else 64
        model = ByteFightNet(hidden_dim=hidden_dim).to(config.device)
        num_actions = 7
    else:
        raise ValueError(f"Unknown game: {config.game}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # Setup checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir) / checkpoint_run_name

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
            None,
            None,
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

    # Build persistent self-play session (threads created but paused).
    replay_buffer = EphemeralReplayBuffer(
        config.replay_buffer_capacity,
        game=config.game,
    )
    if config.game == "bytefight":
        # ByteFight uses the Rust CUDA graph runner for dispatch.
        available_gpus = torch.cuda.device_count()
        if available_gpus < 1:
            raise RuntimeError(
                "bytefight rust-cudagraph self-play requires at least one CUDA GPU"
            )
        num_gpus = config.num_gpus if config.num_gpus > 0 else available_gpus
        if num_gpus > available_gpus:
            raise ValueError(
                f"Requested --num-gpus={num_gpus}, but only {available_gpus} GPUs are visible"
            )
        selfplay = SelfPlay(
            game=config.game,
            replay_buffer=replay_buffer,
            num_threads=config.num_threads,
            workers_per_thread=config.workers_per_thread,
            mcts_num_simulations=config.mcts_num_simulations,
            mcts_c_puct=config.mcts_c_puct,
            mcts_dirichlet_alpha=config.mcts_dirichlet_alpha,
            mcts_dirichlet_epsilon=config.mcts_dirichlet_epsilon,
            temperature=config.selfplay_temperature,
            exploration_moves=config.selfplay_exploration_moves,
            seed=config.seed,
            model=model,
            selfplay_precision=config.selfplay_precision,
            num_gpus=num_gpus,
        )
    else:
        # Other games use a Python callback for inference.
        execute_model = make_execute_model(
            model,
            config.device,
            num_actions,
            use_inference_cuda_graph=config.device.startswith("cuda"),
            selfplay_precision=config.selfplay_precision,
        )
        selfplay = SelfPlay(
            game=config.game,
            replay_buffer=replay_buffer,
            num_threads=config.num_threads,
            workers_per_thread=config.workers_per_thread,
            mcts_num_simulations=config.mcts_num_simulations,
            mcts_c_puct=config.mcts_c_puct,
            mcts_dirichlet_alpha=config.mcts_dirichlet_alpha,
            mcts_dirichlet_epsilon=config.mcts_dirichlet_epsilon,
            temperature=config.selfplay_temperature,
            exploration_moves=config.selfplay_exploration_moves,
            seed=config.seed,
            execute_model=execute_model,
        )

    # Track metrics
    global_step = start_epoch

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()

        # Self-play phase: use absolute sample targets
        model.eval()
        before = selfplay.samples()
        selfplay_start = time.time()
        reached = selfplay.wait_for(before + config.samples_per_epoch)
        selfplay_time = time.time() - selfplay_start
        samples_collected = reached - before

        # Log self-play metrics
        selfplay_metrics = {
            "epoch": epoch,
            "selfplay/samples_collected": samples_collected,
            "selfplay/samples_total": reached,
            "selfplay/mcts_num_simulations": config.mcts_num_simulations,
            "selfplay/mcts_c_puct": config.mcts_c_puct,
            "selfplay/mcts_dirichlet_alpha": config.mcts_dirichlet_alpha,
            "selfplay/mcts_dirichlet_epsilon": config.mcts_dirichlet_epsilon,
            "selfplay/temperature": config.selfplay_temperature,
            "selfplay/exploration_moves": config.selfplay_exploration_moves,
            "selfplay/time_sec": selfplay_time,
            "selfplay/samples_per_sec": (
                samples_collected / selfplay_time if selfplay_time > 0 else 0.0
            ),
            "selfplay/backend": config.selfplay_backend,
            "replay_buffer/size": len(replay_buffer),
        }
        wandb.log(selfplay_metrics, step=global_step)

        # Training phase (selfplay is paused/quiesced, safe to read buffer)
        if len(replay_buffer) >= config.train_batch_size:
            model.train()
            train_start = time.time()

            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_l2_loss = 0.0
            epoch_total_loss = 0.0
            epoch_value_pred_mean = 0.0
            epoch_value_pred_std = 0.0
            epoch_target_value_mean = 0.0
            epoch_target_value_std = 0.0
            epoch_value_sign_acc = 0.0
            epoch_value_corr = 0.0

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
                epoch_value_pred_mean += losses["value_pred_mean"]
                epoch_value_pred_std += losses["value_pred_std"]
                epoch_target_value_mean += losses["target_value_mean"]
                epoch_target_value_std += losses["target_value_std"]
                epoch_value_sign_acc += losses["value_sign_acc"]
                epoch_value_corr += losses["value_corr"]

            train_time = time.time() - train_start
            num_steps = config.train_steps_per_epoch
            train_samples_drawn = num_steps * config.train_batch_size

            # Log training metrics
            train_metrics = {
                "train/policy_loss": epoch_policy_loss / num_steps,
                "train/value_loss": epoch_value_loss / num_steps,
                "train/l2_loss": epoch_l2_loss / num_steps,
                "train/total_loss": epoch_total_loss / num_steps,
                "train/value_pred_mean": epoch_value_pred_mean / num_steps,
                "train/value_pred_std": epoch_value_pred_std / num_steps,
                "train/target_value_mean": epoch_target_value_mean / num_steps,
                "train/target_value_std": epoch_target_value_std / num_steps,
                "train/value_sign_acc": epoch_value_sign_acc / num_steps,
                "train/value_corr": epoch_value_corr / num_steps,
                "train/time_sec": train_time,
                "train/steps_per_sec": num_steps / train_time,
                "train/samples_drawn": train_samples_drawn,
                "train/sample_reuse_ratio_vs_selfplay": (
                    train_samples_drawn / max(samples_collected, 1)
                ),
            }
            wandb.log(train_metrics, step=global_step)

        global_step += 1
        epoch_time = time.time() - epoch_start

        # Print progress
        buffer_pct = 100 * len(replay_buffer) / config.replay_buffer_capacity
        print(
            f"Epoch {epoch}: {samples_collected} samples, "
            f"buffer {len(replay_buffer)}/{config.replay_buffer_capacity} ({buffer_pct:.1f}%), "
            f"{epoch_time:.1f}s"
        )

        # Save checkpoint periodically
        if (epoch + 1) % config.checkpoint_every == 0:
            save_checkpoint(
                model, optimizer, epoch, config, replay_buffer, checkpoint_dir
            )

    selfplay.drop()

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero-style agent")
    parser.add_argument(
        "--game",
        type=str,
        default="bytefight",
        choices=["tictactoe", "connect4", "bytefight"],
        help="Game to train on",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=1_000_000,
        help="Target samples to collect per epoch",
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=2048, help="Training batch size"
    )
    parser.add_argument(
        "--train-steps-per-epoch",
        type=int,
        default=512,
        help="Training steps per epoch",
    )
    parser.add_argument(
        "--replay-buffer-capacity",
        type=int,
        default=10_000_000,
        help="Replay buffer capacity",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=2.0,
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
        "--workers-per-thread", type=int, default=96, help="Workers per thread"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help=(
            "Model width override. Uses game defaults when omitted "
            "(TicTacToe=128, Connect4=64 channels, ByteFight=64)."
        ),
    )
    parser.add_argument(
        "--mcts-num-simulations",
        type=int,
        default=64,
        help="MCTS simulations per move during self-play",
    )
    parser.add_argument(
        "--mcts-c-puct",
        type=float,
        default=1.5,
        help="PUCT exploration constant",
    )
    parser.add_argument(
        "--mcts-dirichlet-alpha",
        type=float,
        default=0.3,
        help="Dirichlet alpha for root noise",
    )
    parser.add_argument(
        "--mcts-dirichlet-epsilon",
        type=float,
        default=0.25,
        help="Root noise mixing weight in [0, 1]",
    )
    parser.add_argument(
        "--selfplay-temperature",
        type=float,
        default=1.0,
        help="Move sampling temperature before greedy phase",
    )
    parser.add_argument(
        "--selfplay-exploration-moves",
        type=int,
        default=30,
        help="Number of opening moves that use sampling temperature",
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
        "--run-name",
        type=str,
        default=None,
        help="Optional explicit W&B run name",
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
        "--selfplay-backend",
        type=str,
        default="rust-cudagraph",
        choices=["python", "rust-cudagraph"],
        help="Self-play inference backend",
    )
    parser.add_argument(
        "--selfplay-precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
        help="Precision for self-play model inference",
    )
    parser.add_argument(
        "--matmul-precision",
        type=str,
        default="high",
        choices=["highest", "high", "medium"],
        help="Set float32 matmul precision (CUDA only)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs for self-play (0=auto-detect via torch.cuda.device_count())",
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
        hidden_dim=args.hidden_dim,
        mcts_num_simulations=args.mcts_num_simulations,
        mcts_c_puct=args.mcts_c_puct,
        mcts_dirichlet_alpha=args.mcts_dirichlet_alpha,
        mcts_dirichlet_epsilon=args.mcts_dirichlet_epsilon,
        selfplay_temperature=args.selfplay_temperature,
        selfplay_exploration_moves=args.selfplay_exploration_moves,
        seed=args.seed,
        device=args.device,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        compile=args.compile,
        compile_mode=args.compile_mode,
        compile_fullgraph=args.compile_fullgraph,
        cudagraphs=args.cudagraphs,
        selfplay_backend=args.selfplay_backend,
        selfplay_precision=args.selfplay_precision,
        matmul_precision=args.matmul_precision,
        num_gpus=args.num_gpus,
    )

    train(config)


if __name__ == "__main__":
    main()

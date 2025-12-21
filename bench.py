import argparse
import os
import time
from dataclasses import dataclass
from typing import Iterable, cast


def _ensure_cuda_env_for_torch_compile() -> None:
    # torch.compile (inductor->triton) tries to discover libcuda via `/sbin/ldconfig`.
    # In Nix shells this binary may not exist, so point Triton at libcuda explicitly.
    if os.environ.get("TRITON_LIBCUDA_PATH"):
        return

    candidates: list[str] = []
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    candidates.extend([p for p in ld_library_path.split(":") if p])
    candidates.append("/run/opengl-driver/lib")

    for p in candidates:
        if os.path.exists(os.path.join(p, "libcuda.so.1")):
            os.environ["TRITON_LIBCUDA_PATH"] = p
            return


def _ensure_cuda_env_for_tinygrad() -> None:
    # tinygrad picks device from env at import time
    # (backend name is "NV" for NVIDIA/CUDA)
    os.environ.setdefault("DEVICE", "NV")

    # tinygrad CUDA JIT uses NVRTC; in this repo's nix shell CUDA libs
    # live under $CUDA_PATH/lib (not just lib64).
    cuda_path = (
        os.environ.get("CUDA_PATH")
        or os.environ.get("CUDA_HOME")
        or os.environ.get("CUDA_ROOT")
    )
    if not cuda_path:
        return

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in ld_library_path.split(":") if p]

    for candidate in (os.path.join(cuda_path, "lib"), os.path.join(cuda_path, "lib64")):
        if os.path.isdir(candidate) and candidate not in parts:
            parts.insert(0, candidate)

    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


_ensure_cuda_env_for_torch_compile()
_ensure_cuda_env_for_tinygrad()

import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
import tempfile

from tinygrad.dtype import dtypes
from tinygrad import nn as tg
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor


# =============================================================================
# AlphaGo-style ResNet for 19x19 Go board (~20M params)
# Based on AlphaGo Zero: 19 residual blocks, 256 filters
# Input: 17 planes (8 history * 2 colors + 1 color-to-play)
# =============================================================================


class ResBlockTorch(nn.Module):
    """Pre-activation residual block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class AlphaGoResNetTorch(nn.Module):
    """
    AlphaGo Zero style network.
    - Input: (B, 17, 19, 19) - 17 input planes on 19x19 board
    - Tower: 1 conv + N residual blocks (256 filters)
    - Policy head: conv -> flatten -> linear -> softmax (362 moves = 19*19 + pass)
    - Value head: conv -> flatten -> linear -> tanh
    """

    def __init__(self, num_blocks: int = 19, channels: int = 256) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(17, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        self.residual_blocks = nn.Sequential(
            *[ResBlockTorch(channels) for _ in range(num_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 19 * 19, 362)  # 361 board + 1 pass

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(19 * 19, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Input block
        out = torch.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        out = self.residual_blocks(out)

        # Policy head
        policy = torch.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        policy = torch.log_softmax(self.policy_fc(policy), dim=1)

        # Value head
        value = torch.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = torch.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


class ResBlockTinygrad:
    """Pre-activation residual block for tinygrad."""

    def __init__(self, channels: int) -> None:
        self.conv1 = tg.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = tg.BatchNorm2d(channels)
        self.conv2 = tg.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = tg.BatchNorm2d(channels)

    def __call__(self, x: Tensor) -> Tensor:
        residual = x
        out = self.bn1(self.conv1(x)).relu()
        out = self.bn2(self.conv2(out))
        return (out + residual).relu()


class AlphaGoResNetTinygrad:
    """AlphaGo Zero style network for tinygrad."""

    def __init__(self, num_blocks: int = 19, channels: int = 256) -> None:
        self.input_conv = tg.Conv2d(17, channels, 3, padding=1, bias=False)
        self.input_bn = tg.BatchNorm2d(channels)

        self.residual_blocks = [ResBlockTinygrad(channels) for _ in range(num_blocks)]

        # Policy head
        self.policy_conv = tg.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = tg.BatchNorm2d(2)
        self.policy_fc = tg.Linear(2 * 19 * 19, 362)

        # Value head
        self.value_conv = tg.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = tg.BatchNorm2d(1)
        self.value_fc1 = tg.Linear(19 * 19, 256)
        self.value_fc2 = tg.Linear(256, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Input block
        out = self.input_bn(self.input_conv(x)).relu()

        # Residual tower
        for block in self.residual_blocks:
            out = block(out)

        # Policy head
        policy = self.policy_bn(self.policy_conv(out)).relu()
        policy = policy.flatten(1)
        policy = self.policy_fc(policy).log_softmax()

        # Value head
        value = self.value_bn(self.value_conv(out)).relu()
        value = value.flatten(1)
        value = self.value_fc1(value).relu()
        value = self.value_fc2(value).tanh()

        return policy, value


class SnakeModelTorch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.board = nn.Sequential(
            nn.Conv2d(7, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )  # (B, 64)

        self.history = nn.Conv1d(18, 32, kernel_size=8)  # (B, 32)
        self.heuristic = nn.Sequential(
            nn.Linear(18, 32), nn.ReLU(), nn.Linear(32, 32)
        )  # (B, 32)

        self.trunk = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
        self.policy_head = nn.Sequential(nn.Linear(32, 28), nn.LogSoftmax(dim=1))
        self.value_head = nn.Sequential(nn.Linear(32, 1), nn.Tanh())

    def forward(
        self,
        board: torch.Tensor,  # (B, 7, 32, 32)
        heuristic_history: torch.Tensor,  # (B, 8, 18)
        heuristic: torch.Tensor,  # (B, 18)
    ):
        board_emb = self.board(board)  # (B, 64)
        history_emb = torch.relu(
            self.history(heuristic_history.transpose(1, 2)).squeeze(-1)
        )

        assert heuristic_history.size(1) >= 1, (
            "heuristic_history must have at least 1 timestep"
        )
        heuristic_emb = self.heuristic(heuristic)  # (B, 32)
        trunk_emb = torch.cat(
            [board_emb, history_emb, heuristic_emb], dim=1
        )  # (B, 128)
        trunk_emb = self.trunk(trunk_emb)  # (B, 32)
        policy = self.policy_head(trunk_emb)  # (B, 28)
        value = self.value_head(trunk_emb)  # (B, 1)
        return policy, value


class SnakeModelTinygrad:
    def __init__(self) -> None:
        self.boardconv = tg.Conv2d(7, 64, 3, padding=1)
        self.history = tg.Conv1d(18, 32, kernel_size=8)
        self.heuristic1 = tg.Linear(18, 32)
        self.heuristic2 = tg.Linear(32, 32)

        self.trunk1 = tg.Linear(128, 64)
        self.trunk2 = tg.Linear(64, 32)

        self.policy_head = tg.Linear(32, 28)
        self.value_head = tg.Linear(32, 1)

    def forward(
        self,
        board: Tensor,  # (B, 7, 32, 32)
        heuristic_history: Tensor,  # (B, 8, 18)
        heuristic: Tensor,  # (B, 18)
    ):
        board = self.boardconv(board).relu().mean((-2, -1), keepdim=True).flatten(1)
        history = self.history(heuristic_history.transpose(1, 2)).squeeze(-1).relu()
        heuristic = self.heuristic1(heuristic).relu()
        heuristic = self.heuristic2(heuristic).relu()

        trunk = self.trunk1(Tensor.cat(board, history, heuristic, dim=1)).relu()
        trunk = self.trunk2(trunk).relu()

        policy = self.policy_head(trunk).log_softmax()
        value = self.value_head(trunk).tanh()

        return policy, value


@dataclass(frozen=True)
class BenchResult:
    name: str
    batch_size: int
    ms_per_iter: float

    @property
    def iters_per_sec(self) -> float:
        return 1000.0 / self.ms_per_iter

    @property
    def samples_per_sec(self) -> float:
        return self.iters_per_sec * self.batch_size


def _parse_batch_sizes(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"unsupported --dtype {dtype!r} (expected fp16|bf16|fp32)")


def _tinygrad_dtype(dtype: str):
    if dtype == "fp16":
        return dtypes.float16
    if dtype == "bf16":
        return dtypes.bfloat16
    if dtype == "fp32":
        return dtypes.float32
    raise ValueError(f"unsupported --dtype {dtype!r} (expected fp16|bf16|fp32)")


def _bench_torch_compiled(
    batch_size: int, *, iters: int, warmup: int, dtype: torch.dtype
) -> BenchResult:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    model = SnakeModelTorch().to(device="cuda", dtype=dtype).eval()
    compiled = torch.compile(model)

    board = torch.randn((batch_size, 7, 32, 32), device="cuda", dtype=dtype)
    heuristic_history = torch.randn((batch_size, 8, 18), device="cuda", dtype=dtype)
    heuristic = torch.randn((batch_size, 18), device="cuda", dtype=dtype)

    if not board.is_cuda:
        raise RuntimeError("torch inputs are not on CUDA")

    with torch.no_grad():
        # compile + warm caches
        _ = compiled(board, heuristic_history, heuristic)
        torch.cuda.synchronize()
        for _i in range(warmup):
            _ = compiled(board, heuristic_history, heuristic)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _i in range(iters):
            _ = compiled(board, heuristic_history, heuristic)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    ms_per_iter = (t1 - t0) * 1000.0 / iters
    return BenchResult("torch.compile", batch_size, ms_per_iter)


def _bench_alphago_torch_compiled(
    batch_size: int, *, iters: int, warmup: int, dtype: torch.dtype
) -> BenchResult:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    model = AlphaGoResNetTorch().to(device="cuda", dtype=dtype).eval()
    compiled = torch.compile(model)

    # 17 input planes: 8 moves history * 2 colors + 1 color to play
    board = torch.randn((batch_size, 17, 19, 19), device="cuda", dtype=dtype)

    with torch.no_grad():
        _ = compiled(board)
        torch.cuda.synchronize()
        for _ in range(warmup):
            _ = compiled(board)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = compiled(board)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    ms_per_iter = (t1 - t0) * 1000.0 / iters
    return BenchResult("alphago torch", batch_size, ms_per_iter)


def _bench_alphago_tinygrad(
    batch_size: int, *, iters: int, warmup: int, dtype
) -> BenchResult:
    model = AlphaGoResNetTinygrad()
    jit_forward = TinyJit(model.forward)

    board = Tensor.randn(batch_size, 17, 19, 19, device="NV", dtype=dtype)

    # First call builds JIT
    policy, value = jit_forward(board)
    Tensor.realize(policy, value)

    for _ in range(warmup):
        policy, value = cast(tuple[Tensor, Tensor], jit_forward(board))
        Tensor.realize(policy, value)

    t0 = time.perf_counter()
    for _ in range(iters):
        policy, value = cast(tuple[Tensor, Tensor], jit_forward(board))
        Tensor.realize(policy, value)
    t1 = time.perf_counter()

    ms_per_iter = (t1 - t0) * 1000.0 / iters
    return BenchResult("alphago tinygrad", batch_size, ms_per_iter)


def _bench_tinygrad_tinyjit(
    batch_size: int, *, iters: int, warmup: int, dtype
) -> BenchResult:
    # Force CUDA tensors; this will error early if tinygrad CUDA is unavailable.
    model = SnakeModelTinygrad()

    # Wrap forward with TinyJit per batch size (JIT caches by shape)
    jit_forward = TinyJit(model.forward)

    board = Tensor.randn(batch_size, 7, 32, 32, device="NV", dtype=dtype)
    heuristic_history = Tensor.randn(batch_size, 8, 18, device="NV", dtype=dtype)
    heuristic = Tensor.randn(batch_size, 18, device="NV", dtype=dtype)

    if str(board.device) != "NV":
        raise RuntimeError(f"tinygrad input device is not NV (CUDA): {board.device!r}")

    # First call builds the TinyJit graph for this shape.
    policy, value = jit_forward(board, heuristic_history, heuristic)
    Tensor.realize(policy, value)

    for _i in range(warmup):
        policy, value = cast(
            tuple[Tensor, Tensor], jit_forward(board, heuristic_history, heuristic)
        )
        Tensor.realize(policy, value)

    t0 = time.perf_counter()
    for _i in range(iters):
        policy, value = cast(
            tuple[Tensor, Tensor], jit_forward(board, heuristic_history, heuristic)
        )
        Tensor.realize(policy, value)
    t1 = time.perf_counter()

    ms_per_iter = (t1 - t0) * 1000.0 / iters
    return BenchResult("tinygrad TinyJit", batch_size, ms_per_iter)


def _numpy_dtype(dtype: str):
    if dtype == "fp16":
        return np.float16
    if dtype == "bf16":
        # numpy doesn't support bf16, fall back to fp16
        return np.float16
    if dtype == "fp32":
        return np.float32
    raise ValueError(f"unsupported dtype {dtype!r}")


def _export_snake_to_onnx(batch_size: int, dtype: torch.dtype) -> str:
    """Export Snake model to ONNX and return path."""
    model = SnakeModelTorch().to(dtype=dtype).eval()
    board = torch.randn(batch_size, 7, 32, 32, dtype=dtype)
    heuristic_history = torch.randn(batch_size, 8, 18, dtype=dtype)
    heuristic = torch.randn(batch_size, 18, dtype=dtype)

    path = tempfile.mktemp(suffix=".onnx")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (board, heuristic_history, heuristic),
            path,
            input_names=["board", "heuristic_history", "heuristic"],
            output_names=["policy", "value"],
            dynamic_axes=None,
            opset_version=18,
        )
    return path


def _bench_snake_onnx_cuda(
    batch_size: int, *, iters: int, warmup: int, dtype: str
) -> BenchResult:
    torch_dtype = _torch_dtype(dtype)
    np_dtype = _numpy_dtype(dtype)

    onnx_path = _export_snake_to_onnx(batch_size, torch_dtype)

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                },
            ),
        ]
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

        actual_provider = session.get_providers()[0]
        if "CUDA" not in actual_provider:
            raise RuntimeError(f"ONNX Runtime not using CUDA: {actual_provider}")

        board = np.random.randn(batch_size, 7, 32, 32).astype(np_dtype)
        heuristic_history = np.random.randn(batch_size, 8, 18).astype(np_dtype)
        heuristic = np.random.randn(batch_size, 18).astype(np_dtype)

        io_binding = session.io_binding()

        board_ort = ort.OrtValue.ortvalue_from_numpy(board, "cuda", 0)
        hh_ort = ort.OrtValue.ortvalue_from_numpy(heuristic_history, "cuda", 0)
        h_ort = ort.OrtValue.ortvalue_from_numpy(heuristic, "cuda", 0)

        io_binding.bind_ortvalue_input("board", board_ort)
        io_binding.bind_ortvalue_input("heuristic_history", hh_ort)
        io_binding.bind_ortvalue_input("heuristic", h_ort)
        io_binding.bind_output("policy", "cuda")
        io_binding.bind_output("value", "cuda")

        # Warmup
        session.run_with_iobinding(io_binding)
        for _ in range(warmup):
            session.run_with_iobinding(io_binding)

        t0 = time.perf_counter()
        for _ in range(iters):
            session.run_with_iobinding(io_binding)
        t1 = time.perf_counter()

        ms_per_iter = (t1 - t0) * 1000.0 / iters
        return BenchResult("onnx-cuda", batch_size, ms_per_iter)

    finally:
        import os

        os.unlink(onnx_path)


def _bench_snake_onnx_tensorrt(
    batch_size: int, *, iters: int, warmup: int, dtype: str
) -> BenchResult:
    torch_dtype = _torch_dtype(dtype)
    np_dtype = _numpy_dtype(dtype)

    onnx_path = _export_snake_to_onnx(batch_size, torch_dtype)

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_fp16_enable": dtype == "fp16",
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "/tmp/trt_cache_snake",
                },
            ),
            ("CUDAExecutionProvider", {"device_id": 0}),
        ]
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

        # Check we're actually using TensorRT
        if "TensorrtExecutionProvider" not in session.get_providers():
            raise RuntimeError("TensorRT not being used")

        board = np.random.randn(batch_size, 7, 32, 32).astype(np_dtype)
        heuristic_history = np.random.randn(batch_size, 8, 18).astype(np_dtype)
        heuristic = np.random.randn(batch_size, 18).astype(np_dtype)

        io_binding = session.io_binding()

        board_ort = ort.OrtValue.ortvalue_from_numpy(board, "cuda", 0)
        hh_ort = ort.OrtValue.ortvalue_from_numpy(heuristic_history, "cuda", 0)
        h_ort = ort.OrtValue.ortvalue_from_numpy(heuristic, "cuda", 0)

        io_binding.bind_ortvalue_input("board", board_ort)
        io_binding.bind_ortvalue_input("heuristic_history", hh_ort)
        io_binding.bind_ortvalue_input("heuristic", h_ort)
        io_binding.bind_output("policy", "cuda")
        io_binding.bind_output("value", "cuda")

        # Extra warmup for TRT engine building
        for _ in range(warmup + 5):
            session.run_with_iobinding(io_binding)

        t0 = time.perf_counter()
        for _ in range(iters):
            session.run_with_iobinding(io_binding)
        t1 = time.perf_counter()

        ms_per_iter = (t1 - t0) * 1000.0 / iters
        return BenchResult("onnx-trt", batch_size, ms_per_iter)

    finally:
        import os

        os.unlink(onnx_path)


def _export_alphago_to_onnx(batch_size: int, dtype: torch.dtype) -> str:
    """Export AlphaGo model to ONNX and return path."""
    model = AlphaGoResNetTorch().to(dtype=dtype).eval()
    dummy_input = torch.randn(batch_size, 17, 19, 19, dtype=dtype)

    path = tempfile.mktemp(suffix=".onnx")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input,),
            path,
            input_names=["board"],
            output_names=["policy", "value"],
            dynamic_axes=None,  # static shape for max perf
            opset_version=18,
        )
    return path


def _bench_alphago_onnx_cuda(
    batch_size: int, *, iters: int, warmup: int, dtype: str
) -> BenchResult:
    torch_dtype = _torch_dtype(dtype)
    np_dtype = _numpy_dtype(dtype)

    # Export model
    onnx_path = _export_alphago_to_onnx(batch_size, torch_dtype)

    try:
        # Create CUDA session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                },
            ),
        ]
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

        # Verify we're on CUDA
        actual_provider = session.get_providers()[0]
        if "CUDA" not in actual_provider:
            raise RuntimeError(f"ONNX Runtime not using CUDA: {actual_provider}")

        # Create input
        board = np.random.randn(batch_size, 17, 19, 19).astype(np_dtype)
        io_binding = session.io_binding()

        # Bind input to CUDA
        board_ortvalue = ort.OrtValue.ortvalue_from_numpy(board, "cuda", 0)
        io_binding.bind_ortvalue_input("board", board_ortvalue)

        # Bind outputs to CUDA
        io_binding.bind_output("policy", "cuda")
        io_binding.bind_output("value", "cuda")

        # Warmup
        session.run_with_iobinding(io_binding)
        for _ in range(warmup):
            session.run_with_iobinding(io_binding)

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(iters):
            session.run_with_iobinding(io_binding)
        t1 = time.perf_counter()

        ms_per_iter = (t1 - t0) * 1000.0 / iters
        return BenchResult("alphago onnx-cuda", batch_size, ms_per_iter)

    finally:
        import os

        os.unlink(onnx_path)


def _bench_alphago_onnx_tensorrt(
    batch_size: int, *, iters: int, warmup: int, dtype: str
) -> BenchResult:
    torch_dtype = _torch_dtype(dtype)
    np_dtype = _numpy_dtype(dtype)

    # Export model
    onnx_path = _export_alphago_to_onnx(batch_size, torch_dtype)

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_fp16_enable": dtype == "fp16",
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "/tmp/trt_cache",
                },
            ),
            ("CUDAExecutionProvider", {"device_id": 0}),
        ]
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

        board = np.random.randn(batch_size, 17, 19, 19).astype(np_dtype)
        io_binding = session.io_binding()

        board_ortvalue = ort.OrtValue.ortvalue_from_numpy(board, "cuda", 0)
        io_binding.bind_ortvalue_input("board", board_ortvalue)
        io_binding.bind_output("policy", "cuda")
        io_binding.bind_output("value", "cuda")

        # Warmup (TensorRT needs more warmup for engine building)
        for _ in range(warmup + 5):
            session.run_with_iobinding(io_binding)

        t0 = time.perf_counter()
        for _ in range(iters):
            session.run_with_iobinding(io_binding)
        t1 = time.perf_counter()

        ms_per_iter = (t1 - t0) * 1000.0 / iters
        return BenchResult("alphago onnx-trt", batch_size, ms_per_iter)

    finally:
        import os

        os.unlink(onnx_path)


def _print_results(results: Iterable[BenchResult]) -> None:
    rows = list(results)
    print("\nSnake Model Results (lower is better):")
    print("batch\timpl\t\t\tms/iter\t\tsamples/s")
    for r in rows:
        print(
            f"{r.batch_size}\t{r.name:20s}\t{r.ms_per_iter:8.3f}\t\t{r.samples_per_sec:,.0f}"
        )

    by_bs: dict[int, dict[str, BenchResult]] = {}
    for r in rows:
        by_bs.setdefault(r.batch_size, {})[r.name] = r

    print("\nComparison vs torch.compile:")
    for bs in sorted(by_bs.keys()):
        torch_r = by_bs[bs].get("torch.compile")
        if not torch_r:
            continue
        print(f"batch {bs}:")
        for name in ["tinygrad TinyJit", "onnx-cuda", "onnx-trt"]:
            other = by_bs[bs].get(name)
            if other:
                ratio = other.ms_per_iter / torch_r.ms_per_iter
                faster_slower = "slower" if ratio > 1 else "faster"
                print(f"  {name:20s} {ratio:6.2f}x {faster_slower}")


def _print_alphago_results(results: Iterable[BenchResult]) -> None:
    rows = list(results)
    print("\nAlphaGo ResNet Results (lower is better):")
    print("batch\timpl\t\t\tms/iter\t\tsamples/s")
    for r in rows:
        print(
            f"{r.batch_size}\t{r.name:20s}\t{r.ms_per_iter:8.3f}\t\t{r.samples_per_sec:,.0f}"
        )

    by_bs: dict[int, dict[str, BenchResult]] = {}
    for r in rows:
        by_bs.setdefault(r.batch_size, {})[r.name] = r

    print("\nComparison vs torch.compile:")
    for bs in sorted(by_bs.keys()):
        torch_r = by_bs[bs].get("alphago torch")
        if not torch_r:
            continue
        print(f"batch {bs}:")
        for name in ["alphago tinygrad", "alphago onnx-cuda", "alphago onnx-trt"]:
            other = by_bs[bs].get(name)
            if other:
                ratio = other.ms_per_iter / torch_r.ms_per_iter
                faster_slower = "slower" if ratio > 1 else "faster"
                print(f"  {name:20s} {ratio:6.2f}x {faster_slower}")


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--batch-sizes",
        default="128,256,512,1024",
        help="comma-separated batch sizes",
    )
    p.add_argument("--iters", type=int, default=200, help="timed iterations")
    p.add_argument("--warmup", type=int, default=50, help="warmup iterations")
    p.add_argument(
        "--dtype",
        default="fp16",
        choices=("fp16", "bf16", "fp32"),
        help="compute dtype",
    )
    p.add_argument(
        "--model",
        default="snake",
        choices=("snake", "alphago", "all"),
        help="model to benchmark",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to torch")

    dev_name = torch.cuda.get_device_name(0)
    print(f"Torch CUDA device: {dev_name}")

    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    torch_dtype = _torch_dtype(args.dtype)
    tg_dtype = _tinygrad_dtype(args.dtype)

    if args.model in ("snake", "all"):
        results: list[BenchResult] = []
        for bs in batch_sizes:
            results.append(
                _bench_torch_compiled(
                    bs, iters=args.iters, warmup=args.warmup, dtype=torch_dtype
                )
            )
            results.append(
                _bench_tinygrad_tinyjit(
                    bs, iters=args.iters, warmup=args.warmup, dtype=tg_dtype
                )
            )
            try:
                results.append(
                    _bench_snake_onnx_cuda(
                        bs, iters=args.iters, warmup=args.warmup, dtype=args.dtype
                    )
                )
            except Exception as e:
                print(f"ONNX CUDA failed for batch {bs}: {e}")
            try:
                results.append(
                    _bench_snake_onnx_tensorrt(
                        bs, iters=args.iters, warmup=args.warmup, dtype=args.dtype
                    )
                )
            except Exception as e:
                print(f"ONNX TensorRT failed for batch {bs}: {e}")
        _print_results(results)

    if args.model in ("alphago", "all"):
        # Print param count
        alphago_model = AlphaGoResNetTorch()
        print(f"\nAlphaGo ResNet params: {_count_params(alphago_model):,}")
        del alphago_model

        results = []
        for bs in batch_sizes:
            results.append(
                _bench_alphago_torch_compiled(
                    bs, iters=args.iters, warmup=args.warmup, dtype=torch_dtype
                )
            )
            results.append(
                _bench_alphago_tinygrad(
                    bs, iters=args.iters, warmup=args.warmup, dtype=tg_dtype
                )
            )
            # ONNX Runtime benchmarks
            try:
                results.append(
                    _bench_alphago_onnx_cuda(
                        bs, iters=args.iters, warmup=args.warmup, dtype=args.dtype
                    )
                )
            except Exception as e:
                print(f"ONNX CUDA failed for batch {bs}: {e}")
            try:
                results.append(
                    _bench_alphago_onnx_tensorrt(
                        bs, iters=args.iters, warmup=args.warmup, dtype=args.dtype
                    )
                )
            except Exception as e:
                print(f"ONNX TensorRT failed for batch {bs}: {e}")
        _print_alphago_results(results)


if __name__ == "__main__":
    main()

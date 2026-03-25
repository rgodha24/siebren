from typing import Optional

import torch
import torch.utils.dlpack as dlpack


def _validate_bytefight_tensors(
    obs_host: torch.Tensor,
    obs_device: torch.Tensor,
    policy_host: torch.Tensor,
    policy_device: torch.Tensor,
    value_host: torch.Tensor,
    value_device: torch.Tensor,
) -> None:
    if obs_host.device.type != "cpu":
        raise ValueError("obs_host must be a CPU tensor")
    if policy_host.device.type != "cpu" or value_host.device.type != "cpu":
        raise ValueError("policy_host/value_host must be CPU tensors")
    if obs_device.device.type != "cuda":
        raise ValueError("obs_device must be a CUDA tensor")
    if policy_device.device.type != "cuda" or value_device.device.type != "cuda":
        raise ValueError("policy_device/value_device must be CUDA tensors")

    if obs_host.dtype != torch.uint8 or obs_device.dtype != torch.uint8:
        raise ValueError("obs_host/obs_device must be uint8")
    if policy_host.dtype != torch.float32 or policy_device.dtype != torch.float32:
        raise ValueError("policy_host/policy_device must be float32")
    if value_host.dtype != torch.float32 or value_device.dtype != torch.float32:
        raise ValueError("value_host/value_device must be float32")

    if obs_host.ndim != 3 or obs_device.ndim != 3:
        raise ValueError("obs tensors must be rank-3")
    if policy_host.ndim != 2 or policy_device.ndim != 2:
        raise ValueError("policy tensors must be rank-2")
    if value_host.ndim != 1 or value_device.ndim != 1:
        raise ValueError("value tensors must be rank-1")

    batch = obs_host.shape[0]
    if obs_host.shape != obs_device.shape:
        raise ValueError("obs_host/obs_device shape mismatch")
    if obs_host.shape[1] != 18 or obs_host.shape[2] != 16:
        raise ValueError("ByteFight obs must be shape (B, 18, 16)")
    if policy_host.shape != policy_device.shape:
        raise ValueError("policy_host/policy_device shape mismatch")
    if policy_host.shape[0] != batch or policy_host.shape[1] != 7:
        raise ValueError("ByteFight policy must be shape (B, 7)")
    if value_host.shape != value_device.shape:
        raise ValueError("value_host/value_device shape mismatch")
    if value_host.shape[0] != batch:
        raise ValueError("ByteFight value must be shape (B,)")


def _autocast_dtype(precision: str) -> Optional[torch.dtype]:
    if precision == "fp32":
        return None
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    raise ValueError(f"Unsupported precision: {precision}")


def capture_bytefight_lane_graph(
    model,
    obs_host_dlpack,
    obs_device_dlpack,
    policy_host_dlpack,
    policy_device_dlpack,
    value_host_dlpack,
    value_device_dlpack,
    stream_handle: int,
    precision: str = "fp32",
    gpu_id: int = 0,
) -> tuple[int, object]:
    obs_host = dlpack.from_dlpack(obs_host_dlpack)
    obs_device = dlpack.from_dlpack(obs_device_dlpack)
    policy_host = dlpack.from_dlpack(policy_host_dlpack)
    policy_device = dlpack.from_dlpack(policy_device_dlpack)
    value_host = dlpack.from_dlpack(value_host_dlpack)
    value_device = dlpack.from_dlpack(value_device_dlpack)

    _validate_bytefight_tensors(
        obs_host,
        obs_device,
        policy_host,
        policy_device,
        value_host,
        value_device,
    )

    model = model.to(f"cuda:{gpu_id}")
    model.eval()
    stream = torch.cuda.ExternalStream(stream_handle)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    dtype = _autocast_dtype(precision)

    def run_step() -> None:
        obs_device.copy_(obs_host, non_blocking=True)
        if dtype is None:
            logits, value = model(obs_device)
        else:
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits, value = model(obs_device)
        probs = torch.softmax(logits, dim=-1)
        policy_device.copy_(probs, non_blocking=True)
        value_device.copy_(value, non_blocking=True)
        policy_host.copy_(policy_device, non_blocking=True)
        value_host.copy_(value_device, non_blocking=True)

    with torch.inference_mode():
        with torch.cuda.stream(stream):
            for _ in range(3):
                run_step()
        torch.cuda.synchronize()

        with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
            run_step()

    graph.instantiate()
    owner = (
        graph,
        model,
        obs_host,
        obs_device,
        policy_host,
        policy_device,
        value_host,
        value_device,
        stream,
    )
    return int(graph.raw_cuda_graph_exec()), owner

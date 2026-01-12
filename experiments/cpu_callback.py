import torch


def cpu_callback(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cpu":
        raise ValueError("expected CPU tensor")
    if x.dtype != torch.float32:
        x = x.float()
    return torch.tanh(x * 0.1) + 0.5

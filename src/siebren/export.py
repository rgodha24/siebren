"""ONNX model exporting with type introspection."""

import inspect
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, get_type_hints

import torch
import torch.nn as nn


def _get_forward_signature(model: nn.Module) -> Dict[str, Any]:
    """Extract input names and type hints from model's forward method."""
    sig = inspect.signature(model.forward)
    hints = get_type_hints(model.forward)

    inputs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        inputs[name] = hints.get(name, torch.Tensor)

    return inputs


def _parse_tensor_comment(model: nn.Module, param_name: str) -> Tuple[int, ...] | None:
    """
    Parse shape comments from forward method source.
    Looks for patterns like: param: torch.Tensor,  # (B, 7, 32, 32)
    """
    try:
        source = inspect.getsource(model.forward)
    except OSError:
        return None

    for line in source.split("\n"):
        if param_name in line and "#" in line:
            comment = line.split("#")[-1].strip()
            if comment.startswith("(") and comment.endswith(")"):
                # Parse (B, 7, 32, 32) -> (1, 7, 32, 32)
                parts = comment[1:-1].split(",")
                shape = []
                for p in parts:
                    p = p.strip()
                    if p in ("B", "batch", "batch_size", "N"):
                        shape.append(1)  # Dynamic batch dim
                    elif p.startswith("seq"):
                        shape.append(1)  # Dynamic sequence dim
                    else:
                        try:
                            shape.append(int(p))
                        except ValueError:
                            shape.append(1)  # Unknown dynamic dim
                return tuple(shape)
    return None


def _create_dummy_inputs(
    model: nn.Module,
    input_shapes: Dict[str, Tuple[int, ...]] | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Create dummy inputs for tracing based on forward signature."""
    forward_inputs = _get_forward_signature(model)
    dummy = {}

    for name in forward_inputs:
        if input_shapes and name in input_shapes:
            shape = input_shapes[name]
        else:
            # Try to parse from comments
            shape = _parse_tensor_comment(model, name)
            if shape is None:
                raise ValueError(
                    f"Cannot infer shape for '{name}'. "
                    f"Provide it via input_shapes={{'{name}': (batch, ...)}}"
                )
        dummy[name] = torch.randn(shape, device=device, dtype=dtype)

    return dummy


def get_models_dir() -> Path:
    """Get the default models directory."""
    # Look for project root (where pyproject.toml is)
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            models_dir = current / "models"
            models_dir.mkdir(exist_ok=True)
            return models_dir
        current = current.parent

    # Fallback to cwd/models
    models_dir = Path.cwd() / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def save_model(
    model: nn.Module,
    name: str,
    epoch: int,
    input_shapes: Dict[str, Tuple[int, ...]] | None = None,
    output_dir: Path | str | None = None,
    opset_version: int = 17,
) -> str:
    """
    Export a PyTorch model to ONNX format using type introspection.

    Args:
        model: The PyTorch model to export
        name: Base name for the model (e.g., "snake")
        epoch: Training epoch number
        input_shapes: Optional dict mapping input names to shapes.
                      If not provided, attempts to parse from forward() comments.
        output_dir: Directory to save models. Defaults to <project>/models/
        opset_version: ONNX opset version

    Returns:
        Path to the saved ONNX model file
    """
    if output_dir is None:
        output_dir = get_models_dir()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for this model
    model_dir = output_dir / name
    model_dir.mkdir(exist_ok=True)

    filename = f"{name}_epoch_{epoch:04d}.onnx"
    filepath = model_dir / filename

    # Get model device and dtype
    try:
        param = next(model.parameters())
        device = param.device
        dtype = param.dtype
    except StopIteration:
        device = torch.device("cpu")
        dtype = torch.float32

    # For ONNX export, we need float32
    model_fp32 = model.float()
    dummy_inputs = _create_dummy_inputs(
        model, input_shapes, device=device, dtype=torch.float32
    )

    # Get input names from signature
    input_names = list(dummy_inputs.keys())

    # Infer output names from a test forward pass
    model.eval()
    with torch.no_grad():
        outputs = model_fp32(**dummy_inputs)

    if isinstance(outputs, torch.Tensor):
        output_names = ["output"]
    elif isinstance(outputs, (tuple, list)):
        output_names = [f"output_{i}" for i in range(len(outputs))]
    else:
        output_names = ["output"]

    # Create dynamic axes for batch dimension
    dynamic_axes = {}
    for name in input_names:
        dynamic_axes[name] = {0: "batch_size"}
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}

    # Export using legacy exporter (more compatible with LSTMs)
    torch.onnx.export(
        model_fp32,
        tuple(dummy_inputs.values()),
        str(filepath),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    # Restore original dtype
    if dtype != torch.float32:
        model.to(dtype=dtype)

    return str(filepath)


def load_model_path(
    name: str, epoch: int | None = None, models_dir: Path | None = None
) -> str:
    """
    Get path to a saved ONNX model.

    Args:
        name: Model name
        epoch: Specific epoch, or None for latest

    Returns:
        Path to the ONNX file
    """
    if models_dir is None:
        models_dir = get_models_dir()

    model_dir = models_dir / name
    if not model_dir.exists():
        raise FileNotFoundError(f"No models found for '{name}'")

    if epoch is not None:
        filepath = model_dir / f"{name}_epoch_{epoch:04d}.onnx"
        if not filepath.exists():
            raise FileNotFoundError(f"Model not found: {filepath}")
        return str(filepath)

    # Find latest
    models = sorted(model_dir.glob(f"{name}_epoch_*.onnx"))
    if not models:
        raise FileNotFoundError(f"No models found in {model_dir}")

    return str(models[-1])

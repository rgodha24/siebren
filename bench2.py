"""
ChessFormer Speed-Optimized Benchmark Suite

Tests optimized model variants across:
- Backends: Eager, torch.compile, ONNX Runtime, TensorRT
- Batch sizes: 256, 512, 1024
- Dtypes: fp32, bf16, fp8, int8

Usage:
    python chessformer_fast.py --quick      # Sanity check
    python chessformer_fast.py --bench      # Full benchmark
    python chessformer_fast.py --compare    # Compare old vs new architecture
"""

import gc
import os
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class FastChessFormerConfig:
    embed_dim: int = 128
    num_heads: int = 4
    num_blocks: int = 4
    ffn_dim: int = 256
    policy_dim: int = 128
    input_channels: int = 112
    num_squares: int = 64
    # Attention variant: "rope", "relative", "linear", "none"
    attention_type: str = "rope"
    # Use 2x2 patches (16 tokens) instead of 64
    use_patches: bool = False
    patch_size: int = 2  # 2x2 = 4 squares per patch, 16 patches total


def get_speed_tiny_config() -> FastChessFormerConfig:
    """~1.5M params, optimized for speed"""
    return FastChessFormerConfig(
        embed_dim=128,
        num_heads=4,
        num_blocks=4,
        ffn_dim=256,
        policy_dim=128,
        attention_type="rope",
        use_patches=False,
    )


def get_speed_small_config() -> FastChessFormerConfig:
    """~4M params"""
    return FastChessFormerConfig(
        embed_dim=192,
        num_heads=6,
        num_blocks=6,
        ffn_dim=384,
        policy_dim=192,
        attention_type="rope",
        use_patches=False,
    )


def get_patched_config() -> FastChessFormerConfig:
    """16 tokens instead of 64 - fastest variant"""
    return FastChessFormerConfig(
        embed_dim=192,
        num_heads=6,
        num_blocks=4,
        ffn_dim=384,
        policy_dim=192,
        attention_type="rope",
        use_patches=True,
        patch_size=2,
    )


# ============================================================================
# Rotary Position Embedding (RoPE)
# ============================================================================


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding - adds positional info with zero attention overhead."""

    def __init__(self, dim: int, max_seq_len: int = 64, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute for all positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        # (seq_len, dim/2) -> (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, seq_len: int):
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ============================================================================
# Relative Position Bias (alternative to RoPE)
# ============================================================================


class RelativePositionBias2D(nn.Module):
    """Learned 2D relative position bias for chess board structure."""

    def __init__(self, num_heads: int, board_size: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.board_size = board_size

        # Relative positions range from -(board_size-1) to +(board_size-1)
        # So we need (2*board_size - 1) entries per dimension
        table_size = 2 * board_size - 1
        self.bias_table = nn.Parameter(torch.zeros(num_heads, table_size, table_size))

        # Precompute relative position indices
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(board_size), torch.arange(board_size), indexing="ij"
            ),
            dim=-1,
        ).reshape(-1, 2)

        # Relative positions between all pairs of squares
        rel_pos = coords.unsqueeze(1) - coords.unsqueeze(0)  # (64, 64, 2)
        rel_pos = rel_pos + (board_size - 1)  # Shift to [0, 2*board_size-2]

        self.register_buffer("rel_pos_h", rel_pos[..., 0].long())
        self.register_buffer("rel_pos_w", rel_pos[..., 1].long())

    def forward(self, batch_size: int) -> torch.Tensor:
        # Index into the bias table
        bias = self.bias_table[:, self.rel_pos_h, self.rel_pos_w]  # (heads, 64, 64)
        return bias.unsqueeze(0).expand(batch_size, -1, -1, -1)


# ============================================================================
# Attention Variants
# ============================================================================


class RoPEAttention(nn.Module):
    """Multi-head attention with RoPE - FlashAttention compatible."""

    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int = 64):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Apply RoPE
        cos, sin = self.rope(N)
        q, k = apply_rope(q, k, cos, sin)

        # FlashAttention via SDPA - no mask needed!
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class RelativeBiasAttention(nn.Module):
    """Multi-head attention with learned 2D relative position bias."""

    def __init__(self, embed_dim: int, num_heads: int, board_size: int = 8):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rel_pos_bias = RelativePositionBias2D(num_heads, board_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Get relative position bias (shared across batch, but must expand)
        bias = self.rel_pos_bias(B)

        # SDPA with bias - may not use FlashAttention but still efficient
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False
        )

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class LinearAttention(nn.Module):
    """O(N) linear attention using ELU feature map."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Feature map for linear attention (ELU + 1 ensures positivity)
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Linear attention: O(N * d^2) instead of O(N^2 * d)
        # k^T @ v -> (heads, head_dim, head_dim)
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        # q @ (k^T @ v) -> (B, heads, N, head_dim)
        qkv = torch.einsum("bhnd,bhde->bhne", q, kv)
        # Normalizer
        normalizer = torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2))

        out = qkv / (normalizer.unsqueeze(-1) + 1e-6)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class SimpleAttention(nn.Module):
    """Vanilla attention without position encoding - baseline."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


# ============================================================================
# Feed-Forward Network
# ============================================================================


class FFN(nn.Module):
    """SwiGLU-style FFN for better quality/speed tradeoff."""

    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, ffn_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SimpleFFN(nn.Module):
    """Standard FFN with Mish activation."""

    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.mish(self.linear1(x)))


# ============================================================================
# Encoder Block
# ============================================================================


class FastEncoderBlock(nn.Module):
    """Optimized encoder block with pre-norm and configurable attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        attention_type: str = "rope",
        num_squares: int = 64,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        # Select attention variant
        board_size = int(num_squares**0.5)
        if attention_type == "rope":
            self.attn = RoPEAttention(embed_dim, num_heads, num_squares)
        elif attention_type == "relative":
            self.attn = RelativeBiasAttention(embed_dim, num_heads, board_size)
        elif attention_type == "linear":
            self.attn = LinearAttention(embed_dim, num_heads)
        else:
            self.attn = SimpleAttention(embed_dim, num_heads)

        self.ffn = SimpleFFN(embed_dim, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# Embedding Layers
# ============================================================================


class SquareEmbedding(nn.Module):
    """Standard per-square embedding (64 tokens)."""

    def __init__(self, input_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_channels, embed_dim)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 112, 8, 8) -> (B, 64, embed_dim)
        x = x.flatten(2).transpose(1, 2)  # (B, 64, 112)
        return self.norm(self.proj(x))


class PatchEmbedding(nn.Module):
    """Patch-based embedding (16 tokens for 2x2 patches)."""

    def __init__(self, input_channels: int, embed_dim: int, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (8 // patch_size) ** 2
        patch_dim = input_channels * patch_size * patch_size

        self.proj = nn.Linear(patch_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 112, 8, 8)
        B, C, H, W = x.shape
        p = self.patch_size

        # Reshape into patches
        x = x.view(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, self.num_patches, -1)

        return self.norm(self.proj(x))


# ============================================================================
# Output Heads
# ============================================================================


class FastPolicyHead(nn.Module):
    """Simplified policy head."""

    def __init__(self, embed_dim: int, num_squares: int = 64, policy_dim: int = 128):
        super().__init__()
        self.num_squares = num_squares
        self.proj = nn.Linear(embed_dim, policy_dim)
        self.q_proj = nn.Linear(policy_dim, policy_dim)
        self.k_proj = nn.Linear(policy_dim, policy_dim)
        self.scale = policy_dim**0.5

        # For promotions (only relevant for 64-square models)
        if num_squares == 64:
            self.promotion = nn.Linear(policy_dim, 4, bias=False)
        else:
            self.promotion = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, embed_dim)
        x = F.mish(self.proj(x))
        q = self.q_proj(x)
        k = self.k_proj(x)

        # Policy logits from Q-K attention pattern
        logits = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if self.promotion is not None and self.num_squares == 64:
            # Promotion handling (simplified)
            promo_k = k[:, -8:, :]
            promo_offsets = self.promotion(promo_k).transpose(-1, -2)
            promo_offsets = promo_offsets[:, :3, :] + promo_offsets[:, 3:4, :]

            n_promo = logits[:, -16:-8, -8:]
            promo_logits = torch.stack(
                [
                    n_promo + promo_offsets[:, 0:1, :],
                    n_promo + promo_offsets[:, 1:2, :],
                    n_promo + promo_offsets[:, 2:3, :],
                ],
                dim=-1,
            )
            promo_logits = promo_logits.view(x.shape[0], 8, 24)

            return torch.cat([logits.flatten(1), promo_logits.flatten(1)], dim=-1)

        return logits.flatten(1)


class FastValueHead(nn.Module):
    """Simplified value head."""

    def __init__(self, embed_dim: int, num_squares: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, embed_dim) -> (B, 3)
        x = x.transpose(1, 2)  # (B, embed_dim, N)
        x = self.pool(x).squeeze(-1)  # (B, embed_dim)
        return self.mlp(x)


# ============================================================================
# Main Model
# ============================================================================


class FastChessFormer(nn.Module):
    """Speed-optimized ChessFormer without Smolgen."""

    def __init__(self, config: FastChessFormerConfig):
        super().__init__()
        self.config = config

        # Determine sequence length based on patch setting
        if config.use_patches:
            self.num_tokens = (8 // config.patch_size) ** 2
            self.embedding = PatchEmbedding(
                config.input_channels, config.embed_dim, config.patch_size
            )
        else:
            self.num_tokens = 64
            self.embedding = SquareEmbedding(config.input_channels, config.embed_dim)

        # Encoder blocks
        self.blocks = nn.ModuleList(
            [
                FastEncoderBlock(
                    config.embed_dim,
                    config.num_heads,
                    config.ffn_dim,
                    config.attention_type,
                    self.num_tokens,
                )
                for _ in range(config.num_blocks)
            ]
        )

        self.norm = nn.LayerNorm(config.embed_dim, eps=1e-6)

        # Output heads
        self.policy_head = FastPolicyHead(
            config.embed_dim, self.num_tokens, config.policy_dim
        )
        self.value_head = FastValueHead(config.embed_dim, self.num_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 112, 8, 8)
        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        value = self.value_head(x)
        policy = self.policy_head(x)

        return value, policy


# ============================================================================
# Quantization Utilities
# ============================================================================


class QuantizationMode(Enum):
    NONE = "none"
    FP8 = "fp8"
    INT8_DYNAMIC = "int8_dynamic"
    INT8_STATIC = "int8_static"


def quantize_model_int8_dynamic(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization."""
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8, inplace=False
    )


def quantize_model_int8_static(
    model: nn.Module, calibration_data: torch.Tensor
) -> nn.Module:
    """Apply static INT8 quantization with calibration."""
    model_prepared = model.cpu().eval()

    # Configure for static quantization
    model_prepared.qconfig = torch.quantization.get_default_qconfig("x86")
    torch.quantization.prepare(model_prepared, inplace=True)

    # Calibrate
    with torch.inference_mode():
        for i in range(min(10, calibration_data.shape[0] // 32)):
            batch = calibration_data[i * 32 : (i + 1) * 32]
            model_prepared(batch)

    # Convert
    torch.quantization.convert(model_prepared, inplace=True)
    return model_prepared


class FP8LinearInference(nn.Module):
    """FP8 linear layer for Hopper/Blackwell GPUs.

    Uses torch._scaled_mm which requires:
    - mat1: row-major (contiguous) with shape (M, K)
    - mat2: column-major (transposed view) with shape (K, N)

    We store weight as (out_features, in_features) contiguous (same as nn.Linear),
    so that weight.T gives a column-major (in_features, out_features) view.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("weight_fp8", None)
        self.register_buffer("weight_scale", None)
        self.register_buffer("bias", None)

        self._has_bias = bias

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FP8LinearInference":
        layer = cls(linear.in_features, linear.out_features, linear.bias is not None)

        with torch.no_grad():
            # linear.weight is (out_features, in_features) - keep this layout
            # When we call weight.T, we get (in_features, out_features) column-major
            # which is exactly what _scaled_mm needs for mat2
            weight = linear.weight.detach().float().contiguous()  # (out, in) row-major

            amax = weight.abs().max().clamp(min=1e-12)
            scale = amax / 448.0

            layer.weight_scale = scale.cuda()
            layer.weight_fp8 = (weight / scale).to(torch.float8_e4m3fn).cuda()

            if linear.bias is not None:
                layer.bias = linear.bias.detach().cuda()

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # Compute scale in float32 (required by _scaled_mm)
        x_amax = x_flat.float().abs().max().clamp(min=1e-12)
        x_scale = (x_amax / 448.0).float()  # Must be float32 scalar
        x_fp8 = (x_flat.float() / x_scale).to(torch.float8_e4m3fn)

        # weight_fp8 is (out_features, in_features) contiguous (row-major)
        # weight_fp8.T is (in_features, out_features) column-major - exactly what we need!
        # _scaled_mm(A, B) computes A @ B where:
        #   A: (M, K) row-major = x_fp8 (batch, in_features)
        #   B: (K, N) column-major = weight_fp8.T (in_features, out_features)
        out = torch._scaled_mm(
            x_fp8,
            self.weight_fp8.T,  # (in_features, out_features) column-major
            scale_a=x_scale,
            scale_b=self.weight_scale,
            out_dtype=x.dtype,
        )

        out = out.view(*orig_shape[:-1], self.out_features)

        if self.bias is not None:
            out = out + self.bias

        return out


def convert_to_fp8(module: nn.Module) -> nn.Module:
    """Convert all Linear layers to FP8.

    Note: FP8 _scaled_mm requires dimensions to be divisible by 16.
    Layers with incompatible dimensions are skipped.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            # Check if dimensions are compatible with FP8 (must be divisible by 16)
            if child.in_features % 16 == 0 and child.out_features % 16 == 0:
                setattr(module, name, FP8LinearInference.from_linear(child))
            # else: keep the original Linear layer
        else:
            convert_to_fp8(child)
    return module


# ============================================================================
# ONNX Export and Runtime
# ============================================================================


def export_to_onnx(
    model: nn.Module,
    save_path: Path,
    batch_size: int = 256,
    opset_version: int = 17,
) -> Path:
    """Export model to ONNX format."""
    model = model.cpu().eval()

    dummy_input = torch.randn(batch_size, 112, 8, 8)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        opset_version=opset_version,
        input_names=["board"],
        output_names=["value", "policy"],
        dynamic_axes={
            "board": {0: "batch_size"},
            "value": {0: "batch_size"},
            "policy": {0: "batch_size"},
        },
    )

    return save_path


class ONNXInferenceSession:
    """Wrapper for ONNX Runtime inference."""

    def __init__(self, model_path: Path, use_gpu: bool = True):
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = os.cpu_count()

        self.session = ort.InferenceSession(
            str(model_path), sess_options=sess_options, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_np = x.cpu().numpy()
        outputs = self.session.run(None, {self.input_name: x_np})
        return torch.from_numpy(outputs[0]), torch.from_numpy(outputs[1])


# ============================================================================
# TensorRT Export (if available)
# ============================================================================


def export_to_tensorrt(
    model: nn.Module,
    save_path: Path,
    batch_size: int = 256,
    precision: str = "fp16",
) -> Optional[Path]:
    """Export to TensorRT if available."""
    try:
        import torch_tensorrt
    except ImportError:
        print("[warn] torch_tensorrt not available, skipping TensorRT export")
        return None

    model = model.cuda().eval()

    inputs = [
        torch_tensorrt.Input(
            shape=[batch_size, 112, 8, 8],
            dtype=torch.float32 if precision == "fp32" else torch.float16,
        )
    ]

    enabled_precisions = {torch.float32}
    if precision == "fp16":
        enabled_precisions.add(torch.float16)
    elif precision == "int8":
        enabled_precisions.add(torch.int8)

    try:
        trt_model = torch_tensorrt.compile(
            model,
            inputs=inputs,
            enabled_precisions=enabled_precisions,
            workspace_size=1 << 30,
            truncate_long_and_double=True,
        )

        torch.jit.save(trt_model, save_path)
        return save_path
    except Exception as e:
        print(f"[warn] TensorRT compilation failed: {e}")
        return None


# ============================================================================
# Benchmark Infrastructure
# ============================================================================


@contextmanager
def cuda_sync_timing():
    """Context manager for accurate CUDA timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield lambda: (end.record(), torch.cuda.synchronize(), start.elapsed_time(end))[
            2
        ]
    else:
        start = time.perf_counter()
        yield lambda: (time.perf_counter() - start) * 1000


def clear_cuda_cache():
    """Aggressively clear CUDA memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


@dataclass
class BenchmarkResult:
    name: str
    backend: str
    dtype: str
    batch_size: int
    avg_time_ms: float
    throughput: float  # samples/sec
    peak_memory_mb: float
    num_params: int


def run_single_benchmark(
    model_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    num_warmup: int = 20,
    num_iters: int = 100,
) -> tuple[float, float]:
    """Run a single benchmark, return (avg_time_ms, peak_memory_mb)."""

    # Warmup
    for _ in range(num_warmup):
        _ = model_fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iters):
            _ = model_fn()
        end_event.record()
        torch.cuda.synchronize()

        total_ms = start_event.elapsed_time(end_event)
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = model_fn()
        total_ms = (time.perf_counter() - start) * 1000
        peak_memory_mb = 0.0

    avg_time_ms = total_ms / num_iters
    return avg_time_ms, peak_memory_mb


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# Main Benchmark Suite
# ============================================================================


class BenchmarkSuite:
    def __init__(
        self,
        batch_sizes: list[int] = [256, 512, 1024],
        num_warmup: int = 20,
        num_iters: int = 100,
    ):
        self.batch_sizes = batch_sizes
        self.num_warmup = num_warmup
        self.num_iters = num_iters
        self.results: list[BenchmarkResult] = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check capabilities
        self.has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.has_fp8 = self._check_fp8_support()

    def _check_fp8_support(self) -> bool:
        """Check if FP8 is supported on this GPU."""
        if not torch.cuda.is_available():
            return False
        try:
            capability = torch.cuda.get_device_capability()
            # FP8 requires SM89 (Ada) or SM90 (Hopper) or higher
            return capability[0] >= 9 or (capability[0] == 8 and capability[1] >= 9)
        except Exception:
            return False

    def benchmark_eager(
        self,
        config: FastChessFormerConfig,
        name: str,
        dtype: torch.dtype,
        dtype_name: str,
    ):
        """Benchmark model in eager mode."""
        for batch_size in self.batch_sizes:
            clear_cuda_cache()

            try:
                model = FastChessFormer(config).to(self.device, dtype).eval()
                num_params = count_parameters(model)
                x = torch.randn(batch_size, 112, 8, 8, device=self.device, dtype=dtype)

                def run():
                    with torch.inference_mode():
                        return model(x)

                avg_time_ms, peak_memory_mb = run_single_benchmark(
                    run, batch_size, self.num_warmup, self.num_iters
                )

                throughput = (batch_size / avg_time_ms) * 1000

                self.results.append(
                    BenchmarkResult(
                        name=name,
                        backend="eager",
                        dtype=dtype_name,
                        batch_size=batch_size,
                        avg_time_ms=avg_time_ms,
                        throughput=throughput,
                        peak_memory_mb=peak_memory_mb,
                        num_params=num_params,
                    )
                )

                del model, x

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.results.append(
                        BenchmarkResult(
                            name=name,
                            backend="eager",
                            dtype=dtype_name,
                            batch_size=batch_size,
                            avg_time_ms=float("inf"),
                            throughput=0.0,
                            peak_memory_mb=0.0,
                            num_params=0,
                        )
                    )
                    clear_cuda_cache()
                else:
                    raise

    def benchmark_compiled(
        self,
        config: FastChessFormerConfig,
        name: str,
        dtype: torch.dtype,
        dtype_name: str,
    ):
        """Benchmark model with torch.compile."""
        for batch_size in self.batch_sizes:
            clear_cuda_cache()

            try:
                model = FastChessFormer(config).to(self.device, dtype).eval()
                num_params = count_parameters(model)

                compiled_model = torch.compile(
                    model,
                    fullgraph=True,
                    mode="max-autotune",
                )

                x = torch.randn(batch_size, 112, 8, 8, device=self.device, dtype=dtype)

                def run():
                    with torch.inference_mode():
                        return compiled_model(x)

                avg_time_ms, peak_memory_mb = run_single_benchmark(
                    run, batch_size, self.num_warmup, self.num_iters
                )

                throughput = (batch_size / avg_time_ms) * 1000

                self.results.append(
                    BenchmarkResult(
                        name=name,
                        backend="compiled",
                        dtype=dtype_name,
                        batch_size=batch_size,
                        avg_time_ms=avg_time_ms,
                        throughput=throughput,
                        peak_memory_mb=peak_memory_mb,
                        num_params=num_params,
                    )
                )

                del model, compiled_model, x

            except Exception as e:
                print(
                    f"[warn] Compiled benchmark failed for {name}/{dtype_name}/{batch_size}: {e}"
                )
                self.results.append(
                    BenchmarkResult(
                        name=name,
                        backend="compiled",
                        dtype=dtype_name,
                        batch_size=batch_size,
                        avg_time_ms=float("inf"),
                        throughput=0.0,
                        peak_memory_mb=0.0,
                        num_params=0,
                    )
                )
                clear_cuda_cache()

    def benchmark_int8_dynamic(self, config: FastChessFormerConfig, name: str):
        """Benchmark with dynamic INT8 quantization."""
        for batch_size in self.batch_sizes:
            clear_cuda_cache()

            try:
                model = FastChessFormer(config).eval()
                model = quantize_model_int8_dynamic(model)
                num_params = count_parameters(model)

                # INT8 quantized models run on CPU
                x = torch.randn(batch_size, 112, 8, 8)

                def run():
                    with torch.inference_mode():
                        return model(x)

                avg_time_ms, _ = run_single_benchmark(
                    run, batch_size, self.num_warmup // 2, self.num_iters // 2
                )

                throughput = (batch_size / avg_time_ms) * 1000

                self.results.append(
                    BenchmarkResult(
                        name=name,
                        backend="int8_dynamic",
                        dtype="int8",
                        batch_size=batch_size,
                        avg_time_ms=avg_time_ms,
                        throughput=throughput,
                        peak_memory_mb=0.0,
                        num_params=num_params,
                    )
                )

                del model, x

            except Exception as e:
                print(f"[warn] INT8 dynamic benchmark failed: {e}")

    def benchmark_fp8(self, config: FastChessFormerConfig, name: str):
        """Benchmark with FP8 quantization (Hopper/Blackwell only)."""
        if not self.has_fp8:
            print("[skip] FP8 not supported on this GPU")
            return

        for batch_size in self.batch_sizes:
            clear_cuda_cache()

            try:
                model = FastChessFormer(config).to(self.device, torch.bfloat16).eval()
                model = convert_to_fp8(model)
                num_params = count_parameters(model)

                x = torch.randn(
                    batch_size, 112, 8, 8, device=self.device, dtype=torch.bfloat16
                )

                def run():
                    with torch.inference_mode():
                        return model(x)

                avg_time_ms, peak_memory_mb = run_single_benchmark(
                    run, batch_size, self.num_warmup, self.num_iters
                )

                throughput = (batch_size / avg_time_ms) * 1000

                self.results.append(
                    BenchmarkResult(
                        name=name,
                        backend="fp8",
                        dtype="fp8",
                        batch_size=batch_size,
                        avg_time_ms=avg_time_ms,
                        throughput=throughput,
                        peak_memory_mb=peak_memory_mb,
                        num_params=num_params,
                    )
                )

                del model, x

            except Exception as e:
                print(f"[warn] FP8 benchmark failed: {e}")

    def benchmark_onnx(self, config: FastChessFormerConfig, name: str):
        """Benchmark with ONNX Runtime."""
        try:
            import onnxruntime
        except ImportError:
            print("[skip] onnxruntime not installed")
            return

        onnx_path = Path(f"/tmp/{name}_model.onnx")

        try:
            model = FastChessFormer(config).eval()
            num_params = count_parameters(model)
            export_to_onnx(model, onnx_path)
            del model

            session = ONNXInferenceSession(onnx_path, use_gpu=torch.cuda.is_available())

            for batch_size in self.batch_sizes:
                x = torch.randn(batch_size, 112, 8, 8)

                def run():
                    return session(x)

                avg_time_ms, _ = run_single_benchmark(
                    run, batch_size, self.num_warmup, self.num_iters
                )

                throughput = (batch_size / avg_time_ms) * 1000

                self.results.append(
                    BenchmarkResult(
                        name=name,
                        backend="onnx",
                        dtype="fp32",
                        batch_size=batch_size,
                        avg_time_ms=avg_time_ms,
                        throughput=throughput,
                        peak_memory_mb=0.0,
                        num_params=num_params,
                    )
                )

        except Exception as e:
            print(f"[warn] ONNX benchmark failed: {e}")
        finally:
            if onnx_path.exists():
                onnx_path.unlink()

    def run_full_benchmark(self):
        """Run the complete benchmark suite."""
        print("=" * 80)
        print("CHESSFORMER SPEED BENCHMARK SUITE")
        print("=" * 80)

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA: {torch.version.cuda}")
            capability = torch.cuda.get_device_capability()
            print(f"Compute Capability: {capability[0]}.{capability[1]}")
        else:
            print("Running on CPU")

        print(f"PyTorch: {torch.__version__}")
        print(f"BF16 Support: {self.has_bf16}")
        print(f"FP8 Support: {self.has_fp8}")
        print(f"Batch sizes: {self.batch_sizes}")
        print("=" * 80)

        configs = {
            "speed_tiny": get_speed_tiny_config(),
            "speed_small": get_speed_small_config(),
            "patched_16tok": get_patched_config(),
        }

        # Also test different attention types
        attention_variants = {
            "rope": "rope",
            "linear": "linear",
            "none": "none",
        }

        dtypes = {"fp32": torch.float32}
        if self.has_bf16:
            dtypes["bf16"] = torch.bfloat16

        # Main benchmarks
        for config_name, config in configs.items():
            print(f"\n{'=' * 80}")
            print(f"CONFIG: {config_name}")
            print(
                f"  embed_dim={config.embed_dim}, heads={config.num_heads}, blocks={config.num_blocks}"
            )
            print(f"  attention={config.attention_type}, patches={config.use_patches}")
            print("=" * 80)

            for dtype_name, dtype in dtypes.items():
                print(f"\n--- {dtype_name.upper()} ---")

                print("  [eager]", end=" ", flush=True)
                self.benchmark_eager(config, config_name, dtype, dtype_name)
                print("done")

                print("  [compiled]", end=" ", flush=True)
                self.benchmark_compiled(config, config_name, dtype, dtype_name)
                print("done")

            print("\n--- INT8 (dynamic) ---")
            print("  [int8_dynamic]", end=" ", flush=True)
            self.benchmark_int8_dynamic(config, config_name)
            print("done")

            if self.has_fp8:
                print("\n--- FP8 ---")
                print("  [fp8]", end=" ", flush=True)
                self.benchmark_fp8(config, config_name)
                print("done")

            print("\n--- ONNX ---")
            print("  [onnx]", end=" ", flush=True)
            self.benchmark_onnx(config, config_name)
            print("done")

        # Test attention variants on tiny config
        print(f"\n{'=' * 80}")
        print("ATTENTION VARIANT COMPARISON (speed_tiny config)")
        print("=" * 80)

        base_config = get_speed_tiny_config()
        for attn_name, attn_type in attention_variants.items():
            config = FastChessFormerConfig(
                embed_dim=base_config.embed_dim,
                num_heads=base_config.num_heads,
                num_blocks=base_config.num_blocks,
                ffn_dim=base_config.ffn_dim,
                policy_dim=base_config.policy_dim,
                attention_type=attn_type,
            )

            name = f"attn_{attn_name}"
            print(f"\n  [{attn_name}]", end=" ", flush=True)
            self.benchmark_compiled(
                config,
                name,
                torch.bfloat16 if self.has_bf16 else torch.float32,
                "bf16" if self.has_bf16 else "fp32",
            )
            print("done")

        self._print_results()

    def _print_results(self):
        """Print formatted results."""
        print("\n" + "=" * 100)
        print("RESULTS")
        print("=" * 100)

        # Group by config
        configs = sorted(set(r.name for r in self.results))

        for config_name in configs:
            config_results = [r for r in self.results if r.name == config_name]
            if not config_results:
                continue

            num_params = config_results[0].num_params
            print(f"\n{config_name} ({num_params / 1e6:.2f}M params)")
            print("-" * 100)
            print(
                f"{'Backend':<15} {'Dtype':<8} {'Batch':<8} {'Time (ms)':<12} {'Throughput':<15} {'Memory (MB)':<12}"
            )
            print("-" * 100)

            for r in sorted(
                config_results, key=lambda x: (x.backend, x.dtype, x.batch_size)
            ):
                if r.throughput > 0:
                    print(
                        f"{r.backend:<15} {r.dtype:<8} {r.batch_size:<8} "
                        f"{r.avg_time_ms:<12.3f} {r.throughput:<15,.0f} {r.peak_memory_mb:<12.1f}"
                    )
                else:
                    print(
                        f"{r.backend:<15} {r.dtype:<8} {r.batch_size:<8} "
                        f"{'OOM/FAIL':<12} {'-':<15} {'-':<12}"
                    )

        # Summary: best throughput per config
        print("\n" + "=" * 100)
        print("SUMMARY: Best throughput per configuration")
        print("=" * 100)

        for config_name in configs:
            config_results = [
                r for r in self.results if r.name == config_name and r.throughput > 0
            ]
            if config_results:
                best = max(config_results, key=lambda x: x.throughput)
                print(
                    f"{config_name}: {best.throughput:,.0f} samples/sec "
                    f"({best.backend}/{best.dtype}/batch={best.batch_size})"
                )


# ============================================================================
# Comparison with Original Model
# ============================================================================


def compare_with_original():
    """Compare optimized model with original Smolgen-based model."""
    print("\n" + "=" * 80)
    print("COMPARISON: Original (with Smolgen) vs Optimized (without)")
    print("=" * 80)

    # Import or define original model here for comparison
    # For now, just compare different attention types

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    batch_size = 512
    num_warmup = 20
    num_iters = 100

    configs = {
        "RoPE (optimized)": get_speed_tiny_config(),
        "Linear Attention": FastChessFormerConfig(
            embed_dim=128,
            num_heads=4,
            num_blocks=4,
            ffn_dim=256,
            policy_dim=128,
            attention_type="linear",
        ),
        "16-token patches": get_patched_config(),
    }

    print(f"\nBatch size: {batch_size}, dtype: {dtype}")
    print("-" * 60)

    for name, config in configs.items():
        clear_cuda_cache()

        model = FastChessFormer(config).to(device, dtype).eval()
        model = torch.compile(model, fullgraph=True, mode="max-autotune")

        x = torch.randn(batch_size, 112, 8, 8, device=device, dtype=dtype)

        def run():
            with torch.inference_mode():
                return model(x)

        avg_time_ms, peak_mem = run_single_benchmark(
            run, batch_size, num_warmup, num_iters
        )
        throughput = (batch_size / avg_time_ms) * 1000

        print(
            f"{name:<25}: {throughput:>10,.0f} samples/sec, {avg_time_ms:.3f} ms, {peak_mem:.0f} MB"
        )

        del model, x


# ============================================================================
# Quick Test
# ============================================================================


def quick_test():
    """Sanity check that all model variants work."""
    print("Running quick test...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    configs = {
        "speed_tiny": get_speed_tiny_config(),
        "speed_small": get_speed_small_config(),
        "patched": get_patched_config(),
    }

    for name, config in configs.items():
        model = FastChessFormer(config).to(device, dtype).eval()
        x = torch.randn(4, 112, 8, 8, device=device, dtype=dtype)

        with torch.inference_mode():
            value, policy = model(x)

        num_params = count_parameters(model)
        print(
            f"{name}: params={num_params / 1e6:.2f}M, value={value.shape}, policy={policy.shape}"
        )

        del model, x

    print("Quick test passed!")


# ============================================================================
# Entry Point
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FastChessFormer Benchmark Suite")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick sanity test only"
    )
    parser.add_argument("--bench", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--compare", action="store_true", help="Compare model variants")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="Number of benchmark iterations"
    )
    args = parser.parse_args()

    if args.quick:
        quick_test()
    elif args.compare:
        quick_test()
        compare_with_original()
    elif args.bench:
        quick_test()
        suite = BenchmarkSuite(
            batch_sizes=args.batch_sizes,
            num_iters=args.iters,
        )
        suite.run_full_benchmark()
    else:
        # Default: run everything
        quick_test()
        print()
        suite = BenchmarkSuite(
            batch_sizes=args.batch_sizes,
            num_iters=args.iters,
        )
        suite.run_full_benchmark()
        compare_with_original()

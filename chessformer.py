"""
ChessFormer Benchmark Script

Tests different model sizes, batch sizes, and precision modes.
Usage: python chessformer_benchmark.py
"""

import os
import time
from dataclasses import dataclass, field, replace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# FP8 Linear Layer for faster inference
# ============================================================================


class FP8Linear(nn.Module):
    """Linear layer that uses FP8 for fast inference on Blackwell/Hopper GPUs."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store weights in bfloat16, quantize on forward
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # FP8 quantized weight cache (populated by quantize_weights)
        self.register_buffer("weight_fp8", None)
        self.register_buffer("weight_scale", None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def quantize_weights(self) -> None:
        """Pre-quantize weights to FP8 for inference."""
        with torch.no_grad():
            # Dynamic per-tensor scaling
            amax = self.weight.float().abs().max()
            # FP8 E4M3 has max value ~448, add eps for numerical stability
            scale = (amax / 448.0).clamp(min=1e-12)
            # Store scale as float32 0-dim tensor on same device
            self.weight_scale = scale.to(torch.float32)
            # Store weight transposed for column-major layout: (in, out) as view of (out, in)
            # _scaled_mm needs B to be column-major, i.e., a transposed view
            # weight is (out_features, in_features)
            # we need (in_features, out_features) column-major = weight.T with stride (1, out_features)
            w_fp8 = (self.weight.float() / scale).to(torch.float8_e4m3fn)
            self.weight_fp8 = w_fp8  # Store as-is, we'll use .T in forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FP8 matmul requires dimensions divisible by 16
        can_use_fp8 = (
            self.weight_fp8 is not None
            and x.is_cuda
            and self.in_features % 16 == 0
            and self.out_features % 16 == 0
        )

        if can_use_fp8:
            # Use FP8 matmul
            orig_shape = x.shape
            orig_dtype = x.dtype
            x_flat = x.view(-1, self.in_features).contiguous().float()

            # Quantize input dynamically
            x_amax = x_flat.abs().max()
            x_scale = (x_amax / 448.0).clamp(min=1e-12).to(torch.float32)
            x_fp8 = (x_flat / x_scale).to(torch.float8_e4m3fn)

            # FP8 matmul: (M, K) @ (K, N) -> (M, N)
            # x_fp8 is row-major (M, K)
            # weight_fp8.T is column-major (K, N) - this is the key!
            out = torch._scaled_mm(
                x_fp8,
                self.weight_fp8.T,  # Column-major view
                scale_a=x_scale,
                scale_b=self.weight_scale,
                out_dtype=orig_dtype,
            )

            out = out.view(*orig_shape[:-1], self.out_features)
            if self.bias is not None:
                out = out + self.bias
            return out
        else:
            # Fallback to regular linear
            return F.linear(x, self.weight, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FP8Linear":
        """Convert a regular Linear layer to FP8Linear, preserving device and dtype."""
        fp8_linear = cls(
            linear.in_features, linear.out_features, linear.bias is not None
        )
        # Move to same device/dtype as source before copying
        fp8_linear = fp8_linear.to(
            device=linear.weight.device, dtype=linear.weight.dtype
        )
        with torch.no_grad():
            fp8_linear.weight.copy_(linear.weight)
            if linear.bias is not None:
                fp8_linear.bias.copy_(linear.bias)  # type: ignore
        return fp8_linear


def convert_to_fp8(module: nn.Module) -> nn.Module:
    """Recursively convert all Linear layers to FP8Linear."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            fp8_linear = FP8Linear.from_linear(child)
            setattr(module, name, fp8_linear)
        else:
            convert_to_fp8(child)
    return module


def quantize_fp8_weights(module: nn.Module) -> None:
    """Pre-quantize all FP8Linear weights for inference."""
    for child in module.modules():
        if isinstance(child, FP8Linear):
            child.quantize_weights()


# ============================================================================
# Smolgen - Dynamic attention bias generation
# ============================================================================


class Smolgen(nn.Module):
    """
    Dynamically generates attention biases based on the current position.
    More parameter-efficient than storing 64x64 learned biases per head.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_channels: int = 32,
        hidden_size: int = 256,
        gen_size: int = 256,
        shared_weight_gen: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.compress = nn.Linear(embed_dim, hidden_channels, bias=False)
        self.dense1 = nn.Linear(hidden_channels * 64, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size, eps=1e-3)
        self.dense2 = nn.Linear(hidden_size, gen_size * num_heads)
        self.ln2 = nn.LayerNorm(gen_size * num_heads, eps=1e-3)

        if shared_weight_gen is not None:
            self.weight_gen = shared_weight_gen
        else:
            self.weight_gen = nn.Linear(gen_size, 64 * 64, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 64, embed_dim) -> (batch, num_heads, 64, 64)"""
        batch_size = x.shape[0]

        compressed = self.compress(x).view(batch_size, -1)
        hidden = F.silu(self.dense1(compressed))
        hidden = self.ln1(hidden)
        gen_from = F.silu(self.dense2(hidden))
        gen_from = self.ln2(gen_from)
        gen_from = gen_from.view(batch_size, self.num_heads, -1)
        out = self.weight_gen(gen_from)

        return out.view(batch_size, self.num_heads, 64, 64)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        smolgen: Smolgen,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale: float = self.head_dim**-0.5

        # Unfused projections (kept for checkpoint compatibility).
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Optional fused projection for inference (3x fewer GEMMs).
        self.qkv_proj: Optional[nn.Linear] = None

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.smolgen = smolgen

    def fuse_qkv(self) -> None:
        """Create a single QKV projection with identical outputs."""
        if self.qkv_proj is not None:
            return

        qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        with torch.no_grad():
            qkv_proj.weight.copy_(
                torch.cat(
                    [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
                    dim=0,
                )
            )
            qkv_proj.bias.copy_(
                torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0)
            )

        self.qkv_proj = qkv_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 64, embed_dim)"""
        batch_size, seq_len, _ = x.shape

        if self.qkv_proj is None:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_bias = self.smolgen(x)

        # Dense per-head bias generally disables FlashAttention, but SDPA still
        # avoids materializing softmax intermediates and compiles well.
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=False,
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        return self.out_proj(attn_output)


class FastMultiHeadAttention(nn.Module):
    """
    Fast attention without Smolgen - uses learned relative position biases.
    This enables FlashAttention since the bias is static (not input-dependent).
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Fused QKV projection from the start
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Learned relative position bias (like ALiBi but learned)
        # For 8x8 board = 64 positions, we store a full 64x64 bias per head
        self.pos_bias = nn.Parameter(torch.zeros(num_heads, 64, 64))
        self._init_pos_bias()

    def _init_pos_bias(self):
        """Initialize with distance-based decay like ALiBi."""
        with torch.no_grad():
            for h in range(self.num_heads):
                slope = 2 ** (-(h + 1) / self.num_heads * 8)
                for i in range(64):
                    for j in range(64):
                        # Manhattan distance on 8x8 board
                        r1, c1 = i // 8, i % 8
                        r2, c2 = j // 8, j % 8
                        dist = abs(r1 - r2) + abs(c1 - c2)
                        self.pos_bias[h, i, j] = -slope * dist

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 64, embed_dim)"""
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Static bias enables FlashAttention (broadcasted across batch)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self.pos_bias,  # (num_heads, 64, 64) broadcasts to (batch, heads, 64, 64)
            dropout_p=0.0,
            is_causal=False,
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.silu(self.linear1(x)))  # SiLU fuses better than Mish


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        smolgen: Smolgen,
        num_blocks: int,
    ):
        super().__init__()
        self.alpha = (2.0 * num_blocks) ** -0.25
        self.attn: MultiHeadAttention = MultiHeadAttention(
            embed_dim, num_heads, smolgen
        )
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-3)
        self.ffn = FeedForward(embed_dim, ffn_dim)
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) * self.alpha
        x = self.ln1(x)
        x = x + self.ffn(x) * self.alpha
        x = self.ln2(x)
        return x


class FastEncoderBlock(nn.Module):
    """Encoder block without Smolgen for maximum inference speed."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, num_blocks: int):
        super().__init__()
        self.alpha = (2.0 * num_blocks) ** -0.25
        self.attn = FastMultiHeadAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-3)
        self.ffn = FeedForward(embed_dim, ffn_dim)
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) * self.alpha
        x = self.ln1(x)
        x = x + self.ffn(x) * self.alpha
        x = self.ln2(x)
        return x


class MaGating(nn.Module):
    """Multiplicative-Additive Gating."""

    def __init__(self, num_squares: int, embed_dim: int):
        super().__init__()
        self.mult_gate = nn.Parameter(torch.ones(num_squares, embed_dim))
        self.add_gate = nn.Parameter(torch.zeros(num_squares, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu(self.mult_gate) + self.add_gate


class Embedding(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        dense_size: int,
        ffn_dim: int,
        num_blocks: int,
    ):
        super().__init__()
        self.preprocess = nn.Linear(64 * 12, 64 * dense_size)
        self.dense_size = dense_size
        self.embed = nn.Linear(input_channels + dense_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-3)
        self.gating = MaGating(64, embed_dim)
        self.alpha = (2.0 * num_blocks) ** -0.25
        self.ffn = FeedForward(embed_dim, ffn_dim)
        self.out_norm = nn.LayerNorm(embed_dim, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 64, 112)"""
        batch_size = x.shape[0]
        piece_info = x[:, :, :12].reshape(batch_size, -1)
        pos_info = self.preprocess(piece_info).view(batch_size, 64, self.dense_size)
        x = torch.cat([x, pos_info], dim=-1)
        x = F.mish(self.embed(x))
        x = self.norm(x)
        x = self.gating(x)
        x = x + self.ffn(x) * self.alpha
        x = self.out_norm(x)
        return x


class PolicyHead(nn.Module):
    def __init__(self, embed_dim: int, policy_dim: int = 1024):
        super().__init__()
        self.tokens = nn.Linear(embed_dim, policy_dim)
        self.q_proj = nn.Linear(policy_dim, policy_dim)
        self.k_proj = nn.Linear(policy_dim, policy_dim)
        self.scale = policy_dim**0.5
        self.promotion = nn.Linear(policy_dim, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 64, embed_dim) -> (batch, 4288)"""
        x = F.mish(self.tokens(x))
        q = self.q_proj(x)
        k = self.k_proj(x)
        qk = torch.matmul(q, k.transpose(-2, -1))

        promotion_keys = k[:, -8:, :]
        promotion_offsets = (
            self.promotion(promotion_keys).transpose(-1, -2) * self.scale
        )
        promotion_offsets = promotion_offsets[:, :3, :] + promotion_offsets[:, 3:4, :]

        n_promo = qk[:, -16:-8, -8:]
        q_promo = (n_promo + promotion_offsets[:, 0:1, :]).unsqueeze(-1)
        r_promo = (n_promo + promotion_offsets[:, 1:2, :]).unsqueeze(-1)
        b_promo = (n_promo + promotion_offsets[:, 2:3, :]).unsqueeze(-1)

        promo_logits = torch.cat([q_promo, r_promo, b_promo], dim=-1)
        promo_logits = promo_logits.view(x.shape[0], 8, 24) / self.scale

        policy_logits = qk / self.scale
        logits = torch.cat([policy_logits.flatten(1), promo_logits.flatten(1)], dim=-1)

        return logits


class ValueHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_channels: int = 128):
        super().__init__()
        self.embed = nn.Linear(embed_dim, hidden_channels)
        self.dense = nn.Linear(hidden_channels * 64, 128)
        self.wdl = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 64, embed_dim) -> (batch, 3)"""
        x = F.mish(self.embed(x)).flatten(1)
        x = F.mish(self.dense(x))
        return self.wdl(x)


# ============================================================================
# Main Model
# ============================================================================


@dataclass
class ChessFormerConfig:
    embed_dim: int = 256
    num_heads: int = 8
    num_blocks: int = 8
    ffn_dim: int = 512
    embed_dense_size: int = 128
    smolgen_hidden_channels: int = 32
    smolgen_hidden_size: int = 128
    smolgen_gen_size: int = 128
    policy_dim: int = 256


class ChessFormer(nn.Module):
    def __init__(self, config: ChessFormerConfig):
        super().__init__()
        self.config = config

        self.embedding = Embedding(
            input_channels=112,
            embed_dim=config.embed_dim,
            dense_size=config.embed_dense_size,
            ffn_dim=config.ffn_dim,
            num_blocks=config.num_blocks,
        )

        shared_weight_gen = nn.Linear(config.smolgen_gen_size, 64 * 64, bias=False)

        self.blocks: nn.ModuleList = nn.ModuleList()
        for _ in range(config.num_blocks):
            smolgen = Smolgen(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                hidden_channels=config.smolgen_hidden_channels,
                hidden_size=config.smolgen_hidden_size,
                gen_size=config.smolgen_gen_size,
                shared_weight_gen=shared_weight_gen,
            )
            self.blocks.append(
                EncoderBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    ffn_dim=config.ffn_dim,
                    smolgen=smolgen,
                    num_blocks=config.num_blocks,
                )
            )

        self.policy_head = PolicyHead(config.embed_dim, config.policy_dim)
        self.value_head = ValueHead(config.embed_dim)

    def fuse_for_inference(self) -> None:
        """Fuse QKV projections in all attention blocks for faster inference."""
        for block in self.blocks:
            block.attn.fuse_qkv()  # type: ignore[union-attr]

    def forward_tokens(self, tokens: torch.Tensor):
        """tokens: (batch, 64, 112) -> (value, policy)"""
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)

        value = self.value_head(x)
        policy = self.policy_head(x)

        return value, policy

    def forward(self, x: torch.Tensor):
        """x: (batch, 112, 8, 8) -> (value, policy)"""
        if x.ndim == 3:
            return self.forward_tokens(x)

        tokens = x.flatten(2).transpose(1, 2).contiguous()
        return self.forward_tokens(tokens)


class FastChessFormer(nn.Module):
    """
    ChessFormer variant optimized for maximum inference speed.
    - No Smolgen (uses learned positional biases instead)
    - Enables FlashAttention
    - Can use FP8 quantization
    """

    def __init__(self, config: ChessFormerConfig):
        super().__init__()
        self.config = config

        self.embedding = Embedding(
            input_channels=112,
            embed_dim=config.embed_dim,
            dense_size=config.embed_dense_size,
            ffn_dim=config.ffn_dim,
            num_blocks=config.num_blocks,
        )

        self.blocks = nn.ModuleList(
            [
                FastEncoderBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    ffn_dim=config.ffn_dim,
                    num_blocks=config.num_blocks,
                )
                for _ in range(config.num_blocks)
            ]
        )

        self.policy_head = PolicyHead(config.embed_dim, config.policy_dim)
        self.value_head = ValueHead(config.embed_dim)

    def to_fp8(self) -> "FastChessFormer":
        """Convert all linear layers to FP8 for faster inference."""
        convert_to_fp8(self)
        return self

    def quantize_for_inference(self) -> None:
        """Pre-quantize FP8 weights for inference."""
        quantize_fp8_weights(self)

    def forward_tokens(self, tokens: torch.Tensor):
        """tokens: (batch, 64, 112) -> (value, policy)"""
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)

        value = self.value_head(x)
        policy = self.policy_head(x)

        return value, policy

    def forward(self, x: torch.Tensor):
        """x: (batch, 112, 8, 8) -> (value, policy)"""
        if x.ndim == 3:
            return self.forward_tokens(x)

        tokens = x.flatten(2).transpose(1, 2).contiguous()
        return self.forward_tokens(tokens)


# ============================================================================
# Model Configs
# ============================================================================


def get_tiny_config() -> ChessFormerConfig:
    """~3M params"""
    return ChessFormerConfig(
        embed_dim=192,
        num_heads=6,
        num_blocks=6,
        ffn_dim=384,
        embed_dense_size=96,
        smolgen_hidden_channels=24,
        smolgen_hidden_size=96,
        smolgen_gen_size=96,
        policy_dim=192,
    )


def get_small_config() -> ChessFormerConfig:
    """~15M params"""
    return ChessFormerConfig(
        embed_dim=384,
        num_heads=12,
        num_blocks=8,
        ffn_dim=768,
        embed_dense_size=192,
        smolgen_hidden_channels=32,
        smolgen_hidden_size=192,
        smolgen_gen_size=192,
        policy_dim=384,
    )


def get_medium_config() -> ChessFormerConfig:
    """~60M params"""
    return ChessFormerConfig(
        embed_dim=512,
        num_heads=16,
        num_blocks=12,
        ffn_dim=1536,
        embed_dense_size=256,
        smolgen_hidden_channels=32,
        smolgen_hidden_size=256,
        smolgen_gen_size=256,
        policy_dim=512,
    )


# ============================================================================
# Benchmark Utils
# ============================================================================


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_flops(config: ChessFormerConfig, batch_size: int) -> int:
    """Rough FLOP estimate for forward pass."""
    embed_dim = config.embed_dim
    num_heads = config.num_heads
    num_blocks = config.num_blocks
    ffn_dim = config.ffn_dim
    seq_len = 64

    # Per block: attention + FFN
    attn_flops = (
        3 * seq_len * embed_dim * embed_dim  # QKV projections
        + 2 * seq_len * seq_len * embed_dim  # attention matmuls
        + seq_len * embed_dim * embed_dim  # output projection
    )
    ffn_flops = 2 * seq_len * embed_dim * ffn_dim

    # Smolgen estimate
    smolgen_flops = (
        seq_len * embed_dim * config.smolgen_hidden_channels
        + seq_len * config.smolgen_hidden_channels * config.smolgen_hidden_size
        + config.smolgen_hidden_size * config.smolgen_gen_size * num_heads
        + num_heads * config.smolgen_gen_size * 64 * 64
    )

    block_flops = attn_flops + ffn_flops + smolgen_flops
    total_flops = num_blocks * block_flops * batch_size

    return int(total_flops)


def benchmark_model(
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    num_warmup: int = 10,
    num_iters: int = 100,
    compile_model: bool = True,
) -> dict:
    """Benchmark model forward pass."""
    fuse = getattr(model, "fuse_for_inference", None)
    if callable(fuse):
        fuse()

    model = model.to(device=device, dtype=dtype)
    model.eval()

    if compile_model:
        try:
            # Note: can't use both mode and options, so we use options with cpp_wrapper=False
            # for Nix compatibility (avoids g++ stdlib.h issues)
            compiled_model = torch.compile(
                model,
                fullgraph=True,
                options={
                    "cpp_wrapper": False,
                    "max_autotune": True,
                    "triton.cudagraphs": True,
                },
            )
        except Exception as e:
            print(
                f"[warn] torch.compile failed ({type(e).__name__}): {e}; falling back to eager"
            )
            compiled_model = model
    else:
        compiled_model = model

    # Create input
    x = torch.randn(batch_size, 112, 8, 8, device=device, dtype=dtype)

    is_cuda = device.type == "cuda"

    # Warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = compiled_model(x)

    if is_cuda:
        torch.cuda.synchronize()

    # Benchmark
    if is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            for _ in range(num_iters):
                _ = compiled_model(x)
        end.record()
        torch.cuda.synchronize()

        total_time_ms = start.elapsed_time(end)
        avg_time_ms = total_time_ms / num_iters
        throughput = (batch_size * num_iters) / (total_time_ms / 1000.0)
    else:
        start_time = time.perf_counter()
        with torch.inference_mode():
            for _ in range(num_iters):
                _ = compiled_model(x)
        end_time = time.perf_counter()

        total_time_s = end_time - start_time
        avg_time_ms = (total_time_s / num_iters) * 1000.0
        throughput = (batch_size * num_iters) / total_time_s

    # Memory
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            _ = compiled_model(x)
        torch.cuda.synchronize()
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak_memory_mb = 0.0

    return {
        "avg_time_ms": avg_time_ms,
        "throughput": throughput,
        "peak_memory_mb": peak_memory_mb,
    }


def run_benchmarks():
    """Run full benchmark suite."""

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print("=" * 80)

    # Model configs
    configs = {
        "tiny": get_tiny_config(),
        "small": get_small_config(),
        "medium": get_medium_config(),
    }

    batch_sizes = [128, 256, 512, 1024, 2048]

    # Precision modes
    dtypes = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    # Check TF32 status
    tf32_enabled = torch.backends.cuda.matmul.allow_tf32
    print(f"TF32 matmul enabled: {tf32_enabled}")

    results = []

    for model_name, config in configs.items():
        print(f"\n{'=' * 80}")
        print(f"Model: {model_name.upper()}")

        model = ChessFormer(config)
        num_params = count_parameters(model)
        print(f"Parameters: {num_params / 1e6:.2f}M")
        print(
            f"Config: embed_dim={config.embed_dim}, heads={config.num_heads}, blocks={config.num_blocks}"
        )
        print("=" * 80)

        for dtype_name, dtype in dtypes.items():
            print(f"\n--- Precision: {dtype_name} ---")
            print(
                f"{'Batch':>8} {'Time (ms)':>12} {'Throughput':>15} {'Memory (MB)':>12} {'TFLOPS':>10}"
            )
            print("-" * 65)

            for batch_size in batch_sizes:
                model = None
                try:
                    # Clear cache
                    torch.cuda.empty_cache()

                    # Enable/disable TF32 based on dtype
                    if dtype_name == "fp32":
                        torch.backends.cuda.matmul.allow_tf32 = False
                        torch.backends.cudnn.allow_tf32 = False
                    else:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True

                    # Fresh model for each run
                    model = ChessFormer(config)

                    result = benchmark_model(
                        model=model,
                        batch_size=batch_size,
                        device=device,
                        dtype=dtype,
                        num_warmup=10,
                        num_iters=50,
                    )

                    flops = estimate_flops(config, batch_size)
                    tflops = (flops / result["avg_time_ms"]) / 1e9  # TFLOPS

                    print(
                        f"{batch_size:>8} {result['avg_time_ms']:>12.3f} {result['throughput']:>15.1f} {result['peak_memory_mb']:>12.1f} {tflops:>10.2f}"
                    )

                    results.append(
                        {
                            "model": model_name,
                            "dtype": dtype_name,
                            "batch_size": batch_size,
                            **result,
                            "tflops": tflops,
                        }
                    )

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"{batch_size:>8} {'OOM':>12}")
                        torch.cuda.empty_cache()
                    else:
                        raise e

                # Clean up
                if model is not None:
                    del model
                torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Best throughput per model/dtype")
    print("=" * 80)

    for model_name in configs.keys():
        print(f"\n{model_name.upper()}:")
        for dtype_name in dtypes.keys():
            model_dtype_results = [
                r
                for r in results
                if r["model"] == model_name and r["dtype"] == dtype_name
            ]
            if model_dtype_results:
                best = max(model_dtype_results, key=lambda x: x["throughput"])
                print(
                    f"  {dtype_name}: {best['throughput']:.1f} pos/s @ batch={best['batch_size']}, {best['tflops']:.2f} TFLOPS"
                )


def run_tf32_comparison():
    """Compare TF32 enabled vs disabled for fp32."""

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device = torch.device("cuda")
    print(f"\n{'=' * 80}")
    print("TF32 COMPARISON (fp32 dtype)")
    print("=" * 80)

    config = get_small_config()
    batch_size = 512

    for tf32_enabled in [False, True]:
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled

        torch.cuda.empty_cache()
        model = ChessFormer(config)

        result = benchmark_model(
            model=model,
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
            num_warmup=10,
            num_iters=50,
        )

        print(
            f"TF32={'ON' if tf32_enabled else 'OFF'}: {result['avg_time_ms']:.3f} ms, {result['throughput']:.1f} pos/s"
        )

        del model
        torch.cuda.empty_cache()


def quick_test():
    """Quick sanity check that model works."""
    print("Running quick test...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_tiny_config()
    model = ChessFormer(config).to(device)

    x = torch.randn(4, 112, 8, 8, device=device)
    value, policy = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Value shape: {value.shape}")  # (4, 3)
    print(f"Policy shape: {policy.shape}")  # (4, 4288)
    print(f"Parameters: {count_parameters(model) / 1e6:.2f}M")
    print("Quick test passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChessFormer Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--tf32", action="store_true", help="Run TF32 comparison only")
    args = parser.parse_args()

    if args.quick:
        quick_test()
    elif args.tf32:
        run_tf32_comparison()
    else:
        quick_test()
        print("\n")
        run_benchmarks()
        run_tf32_comparison()

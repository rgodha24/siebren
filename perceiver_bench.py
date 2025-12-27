# perceiver_bench_latent0_heur.py
"""
Perceiver benchmark for 32x32 board + heuristics injected into latent[0].

Model:
  - Board tokens: (B, C, 32, 32) -> (B, 1024, d)
  - Learned 2D pos embed for 1024 tokens
  - Latents: (B, L, d), learned
  - Heuristics: (B, 18) -> (B, d), then:
        latents[:, 0, :] += heur_embed
    (NO extra heuristics token; sequence length stays exactly L)
  - Perceiver cross-attn: latents attend to board tokens
  - Full self-attn trunk over latents only
  - Heads read latent[0]: policy (11), value (3)

Bench:
  - torch.compile ONLY
  - dtypes: fp16, fp32
  - compare L=32 vs L=64
  - batch sizes default: 256, 512 (configurable)

Usage:
  python perceiver_bench_latent0_heur.py
  python perceiver_bench_latent0_heur.py --batch-sizes 256 512 --iters 200 --warmup 50
  python perceiver_bench_latent0_heur.py --channels 48 --embed-dim 128 --heads 4 --blocks 4

Notes:
  - For fp32, TF32 can greatly change speed. Default is TF32 ON.
    Use --tf32 0 to disable.
"""

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Building blocks
# ----------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CrossAttention(nn.Module):
    """
    Cross-attn: latents query the input tokens.
      latents: (B, L, d)
      inputs:  (B, N, d)
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, latents: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        B, L, d = latents.shape
        _, N, _ = inputs.shape

        q = self.q(latents)
        k, v = self.kv(inputs).chunk(2, dim=-1)

        q = q.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, L, d)
        return self.out(y)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, d)
        return self.out(y)


class PerceiverBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_hidden: int):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = CrossAttention(dim, heads)
        self.n2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_hidden)

    def forward(self, latents: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        latents = latents + self.attn(self.n1(latents), inputs)
        latents = latents + self.mlp(self.n2(latents))
        return latents


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_hidden: int):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, heads)
        self.n2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


# ----------------------------
# Model
# ----------------------------


@dataclass
class ModelConfig:
    channels: int = 24
    heuristics_dim: int = 18
    embed_dim: int = 64
    heads: int = 4
    perceiver_layers: int = 1
    transformer_blocks: int = 1
    mlp_hidden: int = 128
    latents: int = 8  # keep tiny by default for throughput
    token_pool: int = 4  # 1=no pooling, 2=16x16 tokens, 4=8x8 tokens
    pool_type: str = "max"  # max/avg/sum
    actions: int = 11


class Perceiver(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        if 32 % cfg.token_pool != 0:
            raise ValueError("token_pool must divide 32")
        self.token_side = 32 // cfg.token_pool
        token_count = self.token_side * self.token_side

        # (B, 1024, C) -> (B, 1024, d)
        self.board_proj = nn.Linear(cfg.channels, cfg.embed_dim, bias=False)

        # learned 2D absolute pos embedding for 32x32 tokens
        self.pos = nn.Parameter(torch.zeros(1, token_count, cfg.embed_dim))
        nn.init.normal_(self.pos, mean=0.0, std=0.01)

        # learned latents
        self.latents = nn.Parameter(torch.randn(1, cfg.latents, cfg.embed_dim) * 0.02)

        # heuristics -> d (added into latent 0)
        self.heur_proj = nn.Linear(cfg.heuristics_dim, cfg.embed_dim, bias=False)

        self.perceiver = nn.ModuleList(
            [
                PerceiverBlock(cfg.embed_dim, cfg.heads, cfg.mlp_hidden)
                for _ in range(cfg.perceiver_layers)
            ]
        )

        self.trunk = nn.ModuleList(
            [
                TransformerBlock(cfg.embed_dim, cfg.heads, cfg.mlp_hidden)
                for _ in range(cfg.transformer_blocks)
            ]
        )

        self.out_norm = RMSNorm(cfg.embed_dim)

        self.policy_head = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, cfg.actions, bias=True),
        )
        self.value_head = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, 2, bias=True),
        )

    def forward(
        self, board: torch.Tensor, heur: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # board: (B, C, 32, 32), heur: (B, 18)
        B, C, H, W = board.shape
        assert H == 32 and W == 32

        # tokens: (B, 1024, C)
        if self.cfg.token_pool > 1:
            pool = self.cfg.token_pool
            if self.cfg.pool_type == "max":
                board = F.max_pool2d(board, kernel_size=pool, stride=pool, padding=0)
            elif self.cfg.pool_type == "sum":
                board = F.avg_pool2d(
                    board, kernel_size=pool, stride=pool, padding=0
                ) * (pool * pool)
            else:
                board = F.avg_pool2d(board, kernel_size=pool, stride=pool, padding=0)
        tokens = board.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        tokens = self.board_proj(tokens)
        tokens = tokens + self.pos
        latents = self.latents.expand(B, -1, -1)

        # Inject heuristics into latent 0 (no extra token) without in-place mutation
        heur_emb = self.heur_proj(heur).unsqueeze(1)  # (B, 1, d)
        heur_pad = F.pad(heur_emb, (0, 0, 0, self.cfg.latents - 1))
        latents = latents + heur_pad

        # Cross-attn Perceiver stage
        for layer in self.perceiver:
            latents = layer(latents, tokens)

        # Full self-attn trunk over latents only (T = L)
        for block in self.trunk:
            latents = block(latents)

        latents = self.out_norm(latents)
        g = latents[:, 0, :]  # global summary token

        value = self.value_head(g)
        policy = self.policy_head(g)
        return value, policy


# ----------------------------
# Benchmark
# ----------------------------


def set_sdp_prefs():
    # Prefer flash / mem-efficient SDPA; avoid math fallback where possible.
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    except Exception:
        pass


def compile_model(model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
    return torch.compile(model, fullgraph=True, mode=mode)


@torch.no_grad()
def bench(
    compiled: nn.Module,
    board: torch.Tensor,
    heur: torch.Tensor,
    warmup: int,
    iters: int,
    device: torch.device,
):
    if device.type == "cuda":
        for _ in range(warmup):
            _ = compiled(board, heur)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            _ = compiled(board, heur)
        end.record()
        torch.cuda.synchronize()

        total_ms = start.elapsed_time(end)
        avg_ms = total_ms / iters
    else:
        for _ in range(warmup):
            _ = compiled(board, heur)
        start = time.perf_counter()
        for _ in range(iters):
            _ = compiled(board, heur)
        end = time.perf_counter()
        total_ms = (end - start) * 1000.0
        avg_ms = total_ms / iters

    sps = board.shape[0] / (avg_ms / 1000.0)
    return avg_ms, sps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    p.add_argument("--channels", type=int, default=24)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--perceiver-layers", type=int, default=1)
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--mlp-hidden", type=int, default=128)
    p.add_argument("--latents", type=int, nargs="+", default=[8, 16])
    p.add_argument("--token-pool", type=int, default=4)
    p.add_argument(
        "--pool-type", type=str, default="max", choices=["max", "avg", "sum"]
    )
    p.add_argument("--compile-mode", type=str, default="reduce-overhead")
    p.add_argument("--compile", type=int, default=1)

    p.add_argument("--tf32", type=int, default=1)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for --device cuda.")

    device = torch.device(args.device)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("Device: CPU")
    print(f"PyTorch: {torch.__version__}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"iters={args.iters}, warmup={args.warmup}")
    print(
        f"cfg: C={args.channels}, d={args.embed_dim}, heads={args.heads}, "
        f"L_list={args.latents}, token_pool={args.token_pool}, "
        f"pool={args.pool_type}, perceiver_layers={args.perceiver_layers}, blocks={args.blocks}, "
        f"mlp_hidden={args.mlp_hidden}"
    )
    print()

    if device.type == "cuda":
        set_sdp_prefs()

    # TF32 (fp32 only) — usually faster.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    # also influences some matmul heuristics
    try:
        torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    except Exception:
        pass

    latent_sizes = args.latents
    if device.type == "cuda":
        dtype_specs = [("fp16", torch.float16), ("fp32", torch.float32)]
    else:
        dtype_specs = [("fp32", torch.float32)]

    print(
        f"{'L':>4} {'dtype':>6} {'batch':>6} {'avg_ms':>10} "
        f"{'samples/s':>12} {'compile_s':>10}"
    )
    print("-" * 72)

    for L in latent_sizes:
        for dtype_name, dtype in dtype_specs:
            for bs in args.batch_sizes:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                cfg = ModelConfig(
                    channels=args.channels,
                    embed_dim=args.embed_dim,
                    heads=args.heads,
                    perceiver_layers=args.perceiver_layers,
                    transformer_blocks=args.blocks,
                    mlp_hidden=args.mlp_hidden,
                    latents=L,
                    token_pool=args.token_pool,
                    pool_type=args.pool_type,
                )

                model = Perceiver(cfg).to(device=device, dtype=dtype).eval()

                n_params = sum(p.numel() for p in model.parameters())
                print(f"Model params: {n_params:,}")

                board = torch.randn(
                    bs, cfg.channels, 32, 32, device=device, dtype=dtype
                )
                heur = torch.randn(bs, cfg.heuristics_dim, device=device, dtype=dtype)

                # Compile (specializes on shape+dtype)
                t0 = time.perf_counter()
                if args.compile:
                    cm = compile_model(model, mode=args.compile_mode)
                    with torch.inference_mode():
                        _ = cm(board, heur)  # trigger compilation
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    compile_s = time.perf_counter() - t0
                else:
                    cm = model
                    compile_s = 0.0

                with torch.inference_mode():
                    avg_ms, sps = bench(
                        cm, board, heur, args.warmup, args.iters, device
                    )

                print(
                    f"{L:>4} {dtype_name:>6} {bs:>6} {avg_ms:>10.4f} "
                    f"{sps:>12,.0f} {compile_s:>10.2f}"
                )

                del cm, model, board, heur
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

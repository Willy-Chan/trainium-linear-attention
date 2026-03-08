"""
Correctness and performance comparison for fused BASED causal linear attention.

Compares:
  1. Reference PyTorch (full-matrix polynomial attention)
  2. NKI kernel on Trainium (tiled fused polynomial attention)

Both compute: phi(Q@K.T / sqrt(d)) with phi(x) = 1 + x + 0.5*x^2, causally masked,
then normalized attention-weighted sum of V.
"""

import math
import time
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm

from kernels.pytorch_attention import ref_based_attn, solution_nki

device = xm.xla_device()


class TrainiumBasedLinearAttention(nn.Module):
    """Full BASED attention module using polynomial activation (reference path)."""

    def __init__(self, dim: int, heads: int = 8, eps: float = 1e-6):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.eps = eps
        assert dim % heads == 0
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        b, t, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        y = ref_based_attn(q, k, v, eps=self.eps)
        y = y.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(y.to(hidden_states.dtype))


class TrainiumBasedLinearAttentionNKI(nn.Module):
    """Full BASED attention module using NKI kernel."""

    def __init__(self, dim: int, heads: int = 8, eps: float = 1e-6):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.eps = eps
        assert dim % heads == 0
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        b, t, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        y = solution_nki(q, k, v, eps=self.eps)
        y = y.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(y.to(hidden_states.dtype))


# ---------------------------------------------------------------------------
# Kernel-level correctness and performance
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Moderate size so NKI tiled kernel can show speedup (PyTorch does full seq_len^2)
    batch, heads, seq_len, head_dim = 1, 2, 2048, 64
    eps = 1e-6
    warmup_iters = 5
    bench_iters = 50

    sep = "=" * 60
    print(f"Config: batch={batch}, heads={heads}, seq_len={seq_len}, "
          f"head_dim={head_dim}")
    print(f"Benchmark: warmup={warmup_iters}, iters={bench_iters}\n")

    q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device)

    # --- Reference (PyTorch) ---
    y_ref = ref_based_attn(q, k, v, eps=eps)
    torch_xla.sync()

    # --- NKI ---
    nki_available = False
    try:
        y_nki = solution_nki(q, k, v, eps=eps)
        torch_xla.sync()
        nki_available = True
    except Exception as e:
        print(f"NKI kernel not available: {e}")
        print("Skipping NKI correctness and performance.\n")

    # --- Correctness ---
    if nki_available:
        ok = torch.allclose(y_ref, y_nki, rtol=1e-2, atol=1e-2)
        max_diff = (y_ref - y_nki).abs().max().item()
        print(f"Output shapes: ref {y_ref.shape}, NKI {y_nki.shape}")
        if ok:
            print(f"\n{sep}")
            print("  ***  CORRECT: NKI matches reference  ***")
            print(f"  max |ref - nki| = {max_diff:.2e}")
            print(sep)
        else:
            print(f"\n{sep}")
            print("  ***  FAIL: NKI and reference DIFFER  ***")
            print(f"  max |ref - nki| = {max_diff:.2e}")
            print(sep)

    # --- Performance ---
    if nki_available:
        # Warmup reference
        for _ in range(warmup_iters):
            ref_based_attn(q, k, v, eps=eps)
        torch_xla.sync()

        start_ref = time.perf_counter()
        for _ in range(bench_iters):
            ref_based_attn(q, k, v, eps=eps)
        torch_xla.sync()
        elapsed_ref = time.perf_counter() - start_ref
        ref_ms = 1000.0 * elapsed_ref / bench_iters

        # Warmup NKI
        for _ in range(warmup_iters):
            solution_nki(q, k, v, eps=eps)
        torch_xla.sync()

        start_nki = time.perf_counter()
        for _ in range(bench_iters):
            solution_nki(q, k, v, eps=eps)
        torch_xla.sync()
        elapsed_nki = time.perf_counter() - start_nki
        nki_ms = 1000.0 * elapsed_nki / bench_iters

        pct = 100.0 * (elapsed_ref - elapsed_nki) / elapsed_ref

        print(f"\n  --- Performance ({bench_iters} iterations) ---")
        print(f"  Reference (PyTorch) mean time: {ref_ms:.3f} ms")
        print(f"  NKI kernel mean time:          {nki_ms:.3f} ms")
        print(f"  NKI speedup vs reference:      {pct:+.1f}%")
        tokens = batch * seq_len
        print(f"  NKI throughput:                {tokens * bench_iters / elapsed_nki:.0f} tokens/s")
        print(sep)

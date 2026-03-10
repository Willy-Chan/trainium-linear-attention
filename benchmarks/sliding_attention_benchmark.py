"""
Correctness and performance benchmark for sliding window softmax attention.

Compares:
  1. PyTorch reference: standard causal sliding window softmax attention
  2. NKI kernel on Trainium

Both compute: softmax(Q @ K^T / sqrt(d), masked to window W) @ V
"""

import argparse
import math
import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm

from kernels.nki_attention_sliding import get_sliding_window_kernel


DEFAULT_B = 1
DEFAULT_H = 2
DEFAULT_S = 256
DEFAULT_D = 64
DEFAULT_W = 64
DEFAULT_BLOCK = 64
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20


def ref_sliding_window_attn(q, k, v, window_size=64, eps=1e-6):
    """
    PyTorch reference: causal sliding window softmax attention.
    q, k, v: (b, h, t, d)
    Returns: (b, h, t, d)
    """
    b, h, t, d = q.shape
    scale = 1.0 / math.sqrt(d)
    q_s = q.float() * scale

    # Full score matrix
    scores = torch.matmul(q_s, k.float().transpose(-2, -1))  # (b, h, t, t)

    # Build causal + sliding window mask
    row_idx = torch.arange(t, device=q.device).unsqueeze(1)  # (t, 1)
    col_idx = torch.arange(t, device=q.device).unsqueeze(0)  # (1, t)
    # Valid if: col <= row (causal) AND col >= row - window_size + 1 (window)
    causal = col_idx <= row_idx
    window = col_idx >= (row_idx - window_size + 1)
    mask = causal & window  # (t, t)

    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v.float())
    return out.to(q.dtype)


def nki_sliding_window_attn(q, k, v, window_size=64, block_size=64):
    """
    NKI sliding window attention via Trainium kernel.
    q, k, v: (b, h, t, d)
    Returns: (b, h, t, d)
    """
    b, h, t, d = q.shape
    bh = b * h
    scale = 1.0 / math.sqrt(d)

    kernel, SEQ_PAD, D_PAD, KV_SEQ_PAD, mask_np = get_sliding_window_kernel(
        t, d, num_heads=bh, window_size=window_size, block_size=block_size
    )

    q_scaled = (q * scale).float().reshape(bh, t, d)
    k_flat = k.float().reshape(bh, t, d)
    v_flat = v.float().reshape(bh, t, d)
    dev = q.device

    def pad_qkv(x, rows, cols):
        bh_, r, c = x.shape
        if r >= rows and c >= cols:
            return x
        out = torch.zeros(bh_, rows, cols, device=dev, dtype=torch.float32)
        out[:, :r, :c] = x
        return out

    q_pad = pad_qkv(q_scaled, SEQ_PAD, D_PAD)

    # K and V need extra zero rows at the front for window padding
    pad_rows = KV_SEQ_PAD - SEQ_PAD  # (WINDOW_BLOCKS - 1) * BLOCK
    k_pad = torch.zeros(bh, KV_SEQ_PAD, D_PAD, device=dev, dtype=torch.float32)
    k_pad[:, pad_rows:pad_rows + t, :d] = k_flat
    v_pad = torch.zeros(bh, KV_SEQ_PAD, D_PAD, device=dev, dtype=torch.float32)
    v_pad[:, pad_rows:pad_rows + t, :d] = v_flat

    # Additive mask tensor
    mask_tensor = torch.tensor(mask_np, device=dev, dtype=torch.float32)

    out = kernel(q_pad, k_pad, v_pad, mask_tensor)
    return out[:, :t, :d].reshape(b, h, t, d).to(q.dtype)


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch_xla.sync()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch_xla.sync()
    return (time.perf_counter() - start) / iters


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sliding window softmax attention"
    )
    parser.add_argument("--b", type=int, default=DEFAULT_B)
    parser.add_argument("--h", type=int, default=DEFAULT_H)
    parser.add_argument("--s", type=int, default=DEFAULT_S)
    parser.add_argument("--d", type=int, default=DEFAULT_D)
    parser.add_argument("--w", type=int, default=DEFAULT_W,
                        help="Window size (tokens)")
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    args = parser.parse_args()

    device = xm.xla_device()
    torch.manual_seed(0)

    b, h, s, d, w = args.b, args.h, args.s, args.d, args.w
    print(f"Config: B={b}, H={h}, S={s}, D={d}, W={w}, block={args.block}")
    print(f"Warmup={args.warmup}, iters={args.iters}\n")

    q = torch.randn(b, h, s, d, device=device)
    k = torch.randn(b, h, s, d, device=device)
    v = torch.randn(b, h, s, d, device=device)

    # Correctness check
    y_ref = ref_sliding_window_attn(q, k, v, window_size=w)
    y_nki = nki_sliding_window_attn(q, k, v, window_size=w, block_size=args.block)
    torch_xla.sync()

    ok = torch.allclose(y_ref, y_nki, rtol=1e-2, atol=1e-2)
    max_diff = (y_ref - y_nki).abs().max().item()
    sep = "=" * 60
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

    # Performance
    t_ref = bench(
        lambda: ref_sliding_window_attn(q, k, v, window_size=w),
        args.warmup, args.iters
    )
    t_nki = bench(
        lambda: nki_sliding_window_attn(q, k, v, window_size=w, block_size=args.block),
        args.warmup, args.iters
    )

    print(f"\n  --- Performance ({args.iters} iterations) ---")
    print(f"  Reference (PyTorch) mean time: {t_ref * 1000:.3f} ms")
    print(f"  NKI kernel mean time:          {t_nki * 1000:.3f} ms")
    if t_ref > 0:
        pct = 100.0 * (t_ref - t_nki) / t_ref
        print(f"  NKI speedup vs reference:      {pct:+.1f}%")
    toks = b * s
    if t_nki > 0:
        print(f"  NKI throughput:                {toks / t_nki:.0f} tokens/s")
    print(sep)


if __name__ == "__main__":
    main()

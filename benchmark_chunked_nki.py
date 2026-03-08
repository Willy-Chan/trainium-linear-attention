import argparse
import math
import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm

from based_linear_attn_nki_CHUNKED import get_chunked_attn_kernel


# Easy-to-edit defaults
DEFAULT_B = 1
DEFAULT_H = 2
DEFAULT_S = 2048
DEFAULT_D = 64
DEFAULT_CHUNK = 64
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 50
DEFAULT_EPS = 1e-6


def _flatten_diag_outer_product_off1(x):
    """Return strict-upper-triangular and diagonal terms of x x^T."""
    z = torch.einsum("...i,...j->...ij", x, x)
    n = z.size(-1)
    tri_i, tri_j = torch.triu_indices(n, n, 1, device=x.device)
    diag_idx = torch.arange(n, device=x.device, dtype=torch.long)
    x2_1 = z[..., tri_i, tri_j]
    x2_2 = z[..., diag_idx, diag_idx]
    return x2_1, x2_2


def taylor_feature_map(x):
    """
    BASED Taylor feature map.
    Input:  (..., d)
    Output: (..., 1 + d + d + d*(d-1)/2)
    """
    d = x.shape[-1]
    r2 = math.sqrt(2.0)
    rd = math.sqrt(d)
    rrd = math.sqrt(rd)
    x2_1, x2_2 = _flatten_diag_outer_product_off1(x)
    return torch.cat(
        [
            torch.ones_like(x[..., 0:1]),
            x / rrd,
            x2_2 / (rd * r2),
            x2_1 / rd,
        ],
        dim=-1,
    )

def ref_based_attn(q, k, v, eps=1e-6):
    """Dense normalized reference using Taylor feature-mapped Q/K."""
    q_phi = taylor_feature_map(q.float())
    k_phi = taylor_feature_map(k.float())
    scores = torch.matmul(q_phi, k_phi.transpose(-2, -1))
    t = q.shape[-2]
    mask = torch.tril(torch.ones(t, t, device=q.device, dtype=scores.dtype))
    masked = scores * mask
    den = masked.sum(dim=-1, keepdim=True) + eps
    out = torch.matmul(masked, v.float()) / den
    return out.to(q.dtype)


def ref_based_numerator(q, k, v):
    """Dense numerator-only reference using Taylor feature-mapped Q/K."""
    q_phi = taylor_feature_map(q.float())
    k_phi = taylor_feature_map(k.float())
    scores = torch.matmul(q_phi, k_phi.transpose(-2, -1))
    t = q.shape[-2]
    mask = torch.tril(torch.ones(t, t, device=q.device, dtype=scores.dtype))
    out = torch.matmul(scores * mask, v.float())
    return out.to(q.dtype)


# TODO: THIS IS REALLY TESTING IDENTITY NOT BASED
# TODO: BENCHMARK IDENTITY AS WELL!!!!
def ref_identity_numerator(q, k, v):
    """Dense numerator-only reference for pure Identity Linear Attention."""
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    # 1. Scaled Q @ K^T
    scores = torch.matmul(q * scale, k.transpose(-2, -1))
    
    # 2. NO Taylor polynomial! Just apply the mask.
    t = q.shape[-2]
    mask = torch.tril(torch.ones(t, t, device=q.device, dtype=q.dtype))
    
    # 3. Multiply by V
    return torch.matmul(scores * mask, v)


def chunked_nki_numerator(q, k, v, chunk_size=64):
    """
    Run chunked NKI kernel.
    q, k, v: (b, h, t, d)
    returns numerator-only output: (b, h, t, d)
    """
    b, h, t, d = q.shape
    bh = b * h
    q_phi = taylor_feature_map(q.float())
    k_phi = taylor_feature_map(k.float())
    feat_dim = q_phi.shape[-1]
    kernel, seq_pad, feat_pad, v_pad = get_chunked_attn_kernel(
        t, feat_dim, d, chunk_size=chunk_size
    )

    q_flat = q_phi.reshape(bh, t, feat_dim)
    k_flat = k_phi.reshape(bh, t, feat_dim)
    v_flat = v.float().reshape(bh, t, d)

    def pad(x, rows, cols):
        out = torch.zeros(bh, rows, cols, device=x.device, dtype=torch.float32)
        out[:, : x.shape[1], : x.shape[2]] = x
        return out

    q_pad = pad(q_flat, seq_pad, feat_pad)
    k_pad = pad(k_flat, seq_pad, feat_pad)
    v_pad_t = pad(v_flat, seq_pad, v_pad)

    outs = []
    for i in range(bh):
        out_i = kernel(q_pad[i], k_pad[i], v_pad_t[i])
        outs.append(out_i[:t, :d])
    return torch.stack(outs, dim=0).reshape(b, h, t, d).to(q.dtype)


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch_xla.sync()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch_xla.sync()
    elapsed = time.perf_counter() - start
    return elapsed / iters


def main():
    parser = argparse.ArgumentParser(description="Benchmark dense reference vs chunked NKI kernel")
    parser.add_argument("--b", type=int, default=DEFAULT_B)
    parser.add_argument("--h", type=int, default=DEFAULT_H)
    parser.add_argument("--s", type=int, default=DEFAULT_S)
    parser.add_argument("--d", type=int, default=DEFAULT_D)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    args = parser.parse_args()

    device = xm.xla_device()
    torch.manual_seed(0)

    b, h, s, d = args.b, args.h, args.s, args.d
    q = torch.randn(b, h, s, d, device=device)
    k = torch.randn(b, h, s, d, device=device)
    v = torch.randn(b, h, s, d, device=device)

    print(f"Config: B={b}, H={h}, S={s}, D={d}, chunk={args.chunk}")
    print(f"Warmup={args.warmup}, iters={args.iters}")
    print("Note: NKI kernel returns numerator-only output (no denominator normalization).\n")

    # Correctness: compare numerator-only path
    y_ref_num = ref_based_numerator(q, k, v)
    y_nki_num = chunked_nki_numerator(q, k, v, chunk_size=args.chunk)
    torch_xla.sync()
    num_ok = torch.allclose(y_ref_num, y_nki_num, rtol=1e-3, atol=1e-3)
    num_max_diff = (y_ref_num - y_nki_num).abs().max().item()
    print(f"Numerator correctness: {num_ok}  max|diff|={num_max_diff:.3e}")

    # Benchmarks
    t_ref_full = bench(lambda: ref_based_attn(q, k, v, eps=args.eps), args.warmup, args.iters)
    t_ref_num = bench(lambda: ref_based_numerator(q, k, v), args.warmup, args.iters)
    t_nki_num = bench(lambda: chunked_nki_numerator(q, k, v, chunk_size=args.chunk), args.warmup, args.iters)

    print("\n--- Mean time per iteration ---")
    print(f"Dense reference (normalized): {t_ref_full * 1000:.3f} ms")
    print(f"Dense reference (numerator):  {t_ref_num * 1000:.3f} ms")
    print(f"NKI chunked (numerator):      {t_nki_num * 1000:.3f} ms")

    if t_nki_num > 0:
        speedup_vs_num = 100.0 * (t_ref_num - t_nki_num) / t_ref_num
        print(f"NKI speedup vs dense numerator: {speedup_vs_num:+.1f}%")

    toks = b * s
    print(f"NKI throughput: {(toks / t_nki_num):.1f} tokens/s")


if __name__ == "__main__":
    main()

"""
Comprehensive crossover benchmark: NKI kernels vs PyTorch across sequence lengths.

Measures both XLA (lazy) and traced (torch_neuronx.trace) dispatch paths
to isolate dispatch overhead from true kernel performance.
"""

import argparse
import time
import math

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_neuronx

from kernels.pytorch_attention import ref_based_attn, solution_nki
from benchmarks.sliding_attention_benchmark import ref_sliding_window_attn, nki_sliding_window_attn


def bench_xla(fn, device, warmup=5, iters=20):
    """Benchmark through XLA lazy dispatch."""
    for _ in range(warmup):
        fn()
    torch_xla.sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch_xla.sync()
    return 1000 * (time.perf_counter() - start) / iters


def bench_traced(traced_model, args, warmup=10, iters=50):
    """Benchmark a traced model."""
    for _ in range(warmup):
        traced_model(*args)
    torch_xla.sync()
    start = time.perf_counter()
    for _ in range(iters):
        traced_model(*args)
    torch_xla.sync()
    return 1000 * (time.perf_counter() - start) / iters


def main():
    parser = argparse.ArgumentParser(description="Crossover benchmark")
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--h", type=int, default=2)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--w", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--mode", choices=["xla", "traced", "both"], default="both")
    args = parser.parse_args()

    device = xm.xla_device()
    torch.manual_seed(0)
    b, h, d, w = args.b, args.h, args.d, args.w

    seq_lens = [256, 512, 1024, 2048, 4096, 8192, 16384]

    print(f"Config: B={b}, H={h}, D={d}, W={w}")
    print(f"Warmup={args.warmup}, iters={args.iters}\n")

    # ===== XLA (lazy) dispatch =====
    if args.mode in ("xla", "both"):
        print("=" * 80)
        print("XLA DISPATCH (lazy evaluation)")
        print("=" * 80)
        print(f"{'S':>7s}  {'PyT_par':>8s}  {'NKI_par':>8s}  {'par_x':>6s}  "
              f"{'PyT_slid':>9s}  {'NKI_slid':>9s}  {'slid_x':>7s}")
        print("-" * 80)

        for s in seq_lens:
            q = torch.randn(b, h, s, d, device=device)
            k = torch.randn(b, h, s, d, device=device)
            v = torch.randn(b, h, s, d, device=device)

            # Parallel kernel
            t_pyt_par = bench_xla(lambda: ref_based_attn(q, k, v), device,
                                  warmup=args.warmup, iters=args.iters)
            t_nki_par = bench_xla(lambda: solution_nki(q, k, v), device,
                                  warmup=args.warmup, iters=args.iters)
            par_speedup = t_pyt_par / t_nki_par

            # Sliding window
            t_pyt_slid = bench_xla(
                lambda: ref_sliding_window_attn(q, k, v, window_size=w), device,
                warmup=args.warmup, iters=args.iters)
            t_nki_slid = bench_xla(
                lambda: nki_sliding_window_attn(q, k, v, window_size=w), device,
                warmup=args.warmup, iters=args.iters)
            slid_speedup = t_pyt_slid / t_nki_slid

            print(f"{s:7d}  {t_pyt_par:7.2f}ms  {t_nki_par:7.2f}ms  {par_speedup:5.2f}x  "
                  f"{t_pyt_slid:8.2f}ms  {t_nki_slid:8.2f}ms  {slid_speedup:6.2f}x")

    # ===== Traced dispatch =====
    if args.mode in ("traced", "both"):
        print("\n" + "=" * 80)
        print("TRACED DISPATCH (torch_neuronx.trace — reduced overhead)")
        print("=" * 80)
        print(f"{'S':>7s}  {'PyT_par':>8s}  {'NKI_par':>8s}  {'par_x':>6s}  "
              f"{'PyT_slid':>9s}  {'NKI_slid':>9s}  {'slid_x':>7s}")
        print("-" * 80)

        for s in seq_lens:
            q_cpu = torch.randn(b, h, s, d)
            k_cpu = torch.randn(b, h, s, d)
            v_cpu = torch.randn(b, h, s, d)
            q_d = q_cpu.to(device)
            k_d = k_cpu.to(device)
            v_d = v_cpu.to(device)

            # Trace parallel models
            class ParNKI(torch.nn.Module):
                def forward(self, q, k, v):
                    return solution_nki(q, k, v)

            class ParRef(torch.nn.Module):
                def forward(self, q, k, v):
                    return ref_based_attn(q, k, v)

            class SlidNKI(torch.nn.Module):
                def forward(self, q, k, v):
                    return nki_sliding_window_attn(q, k, v, window_size=w)

            class SlidRef(torch.nn.Module):
                def forward(self, q, k, v):
                    return ref_sliding_window_attn(q, k, v, window_size=w)

            try:
                tr_par_nki = torch_neuronx.trace(ParNKI(), (q_cpu, k_cpu, v_cpu))
                t_nki_par = bench_traced(tr_par_nki, (q_d, k_d, v_d),
                                         warmup=args.warmup, iters=args.iters)
            except Exception as e:
                t_nki_par = float('inf')

            try:
                tr_par_ref = torch_neuronx.trace(ParRef(), (q_cpu, k_cpu, v_cpu))
                t_pyt_par = bench_traced(tr_par_ref, (q_d, k_d, v_d),
                                         warmup=args.warmup, iters=args.iters)
            except Exception as e:
                t_pyt_par = float('inf')

            par_speedup = t_pyt_par / t_nki_par if t_nki_par > 0 else 0

            try:
                tr_slid_nki = torch_neuronx.trace(SlidNKI(), (q_cpu, k_cpu, v_cpu))
                t_nki_slid = bench_traced(tr_slid_nki, (q_d, k_d, v_d),
                                          warmup=args.warmup, iters=args.iters)
            except Exception as e:
                t_nki_slid = float('inf')

            try:
                tr_slid_ref = torch_neuronx.trace(SlidRef(), (q_cpu, k_cpu, v_cpu))
                t_pyt_slid = bench_traced(tr_slid_ref, (q_d, k_d, v_d),
                                          warmup=args.warmup, iters=args.iters)
            except Exception as e:
                t_pyt_slid = float('inf')

            slid_speedup = t_pyt_slid / t_nki_slid if t_nki_slid > 0 else 0

            print(f"{s:7d}  {t_pyt_par:7.2f}ms  {t_nki_par:7.2f}ms  {par_speedup:5.2f}x  "
                  f"{t_pyt_slid:8.2f}ms  {t_nki_slid:8.2f}ms  {slid_speedup:6.2f}x")


if __name__ == "__main__":
    main()

"""
Baremetal NKI kernel benchmarks — bypasses XLA framework overhead.

Uses nki.benchmark to run kernels directly on NeuronCore, measuring
true on-device latency without any framework dispatch costs.
"""

import argparse
import math
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


# ============================================================
# Parallel kernel (BASED polynomial attention)
# ============================================================
def bench_parallel(seq_len, dim, num_heads, block_size=64, warmup=10, iters=100):
    BLOCK = block_size
    BH = num_heads
    SEQ_PAD = ((seq_len + BLOCK - 1) // BLOCK) * BLOCK
    D_PAD = ((dim + BLOCK - 1) // BLOCK) * BLOCK
    if D_PAD < 128:
        D_PAD = 128
    NUM_BLOCKS = SEQ_PAD // BLOCK

    _ones_bb = np.ones((BLOCK, BLOCK), dtype=np.float32)
    _lower_mask = np.tril(_ones_bb)

    i_seq = nl.arange(BLOCK)[:, None]
    i_dim = nl.arange(D_PAD)[None, :]
    i_one = nl.arange(1)[None, :]

    @nki.benchmark(warmup=warmup, iters=iters)
    def kernel(q_scaled, k_input, v_input):
        num_out = nl.ndarray((BH, SEQ_PAD, D_PAD), dtype=nl.float32, buffer=nl.hbm, name="num_out")
        den_out = nl.ndarray((BH, SEQ_PAD, 1), dtype=nl.float32, buffer=nl.hbm, name="den_out")
        c_ones = nl.shared_constant(_ones_bb, dtype=nl.float32)
        c_mask = nl.shared_constant(_lower_mask, dtype=nl.float32)

        for bh in nl.affine_range(BH):
            for i in nl.affine_range(NUM_BLOCKS):
                q_off = i * BLOCK
                q_block = nl.load(q_scaled[bh, q_off + i_seq, i_dim])
                acc_num = nl.zeros((BLOCK, D_PAD), dtype=nl.float32, buffer=nl.psum)
                acc_den = nl.zeros((BLOCK, 1), dtype=nl.float32, buffer=nl.psum)

                for j in nl.affine_range(i):
                    k_off = j * BLOCK
                    k_t = nl.load_transpose2d(k_input[bh, k_off + i_seq, i_dim])
                    v_j = nl.load(v_input[bh, k_off + i_seq, i_dim])
                    sc = nl.copy(nl.matmul(q_block, k_t))
                    sq = nl.multiply(sc, sc)
                    poly = nl.add(nl.add(nl.load(c_ones), sc), nl.multiply(sq, 0.5))
                    acc_num += nl.matmul(poly, v_j)
                    acc_den += nl.sum(poly, axis=[1], keepdims=True)

                k_off = i * BLOCK
                k_t = nl.load_transpose2d(k_input[bh, k_off + i_seq, i_dim])
                v_j = nl.load(v_input[bh, k_off + i_seq, i_dim])
                sc = nl.copy(nl.matmul(q_block, k_t))
                sq = nl.multiply(sc, sc)
                poly = nl.multiply(
                    nl.add(nl.add(nl.load(c_ones), sc), nl.multiply(sq, 0.5)),
                    nl.load(c_mask),
                )
                acc_num += nl.matmul(poly, v_j)
                acc_den += nl.sum(poly, axis=[1], keepdims=True)
                nl.store(num_out[bh, q_off + i_seq, i_dim], value=nl.copy(acc_num))
                nl.store(den_out[bh, q_off + i_seq, i_one], value=nl.copy(acc_den))
        return num_out, den_out

    q = np.random.randn(BH, SEQ_PAD, D_PAD).astype(np.float32)
    k = np.random.randn(BH, SEQ_PAD, D_PAD).astype(np.float32)
    v = np.random.randn(BH, SEQ_PAD, D_PAD).astype(np.float32)
    print(f"\n=== Parallel kernel: BH={BH}, S={seq_len}(pad={SEQ_PAD}), D={dim}(pad={D_PAD}) ===")
    kernel(q, k, v)


# ============================================================
# Chunked kernel (recurrent BASED)
# ============================================================
def bench_chunked(seq_len, feat_dim, value_dim, num_heads, chunk_size=64, warmup=10, iters=100):
    BLOCK = chunk_size
    BH = num_heads
    TILE_F = 128

    SEQ_PAD = ((seq_len + BLOCK - 1) // BLOCK) * BLOCK
    FEAT_PAD = ((feat_dim + TILE_F - 1) // TILE_F) * TILE_F
    V_PAD = ((value_dim + BLOCK - 1) // BLOCK) * BLOCK
    if V_PAD < 64:
        V_PAD = 64
    NUM_CHUNKS = SEQ_PAD // BLOCK
    NUM_F_TILES = FEAT_PAD // TILE_F

    _ones_bb = np.ones((BLOCK, BLOCK), dtype=np.float32)
    _lower_mask = np.tril(_ones_bb)
    _zeros_fv = np.zeros((TILE_F, V_PAD), dtype=np.float32)

    i_seq_b = nl.arange(BLOCK)[:, None]
    i_feat_p = nl.arange(TILE_F)[:, None]
    i_feat = nl.arange(TILE_F)[None, :]
    i_vdim = nl.arange(V_PAD)[None, :]

    @nki.benchmark(warmup=warmup, iters=iters)
    def kernel(q_input, k_input, v_input):
        out = nl.ndarray((BH, SEQ_PAD, V_PAD), dtype=nl.float32, buffer=nl.hbm, name="out")
        state = nl.ndarray((BH, FEAT_PAD, V_PAD), dtype=nl.float32, buffer=nl.hbm, name="state")
        c_mask = nl.shared_constant(_lower_mask, dtype=nl.float32)
        c_zero_fv = nl.shared_constant(_zeros_fv, dtype=nl.float32)

        for bh in nl.affine_range(BH):
            for f_idx in nl.affine_range(NUM_F_TILES):
                f_off = f_idx * TILE_F
                nl.store(state[bh, f_off + i_feat_p, i_vdim], value=nl.load(c_zero_fv))

            for c in nl.sequential_range(NUM_CHUNKS):
                off = c * BLOCK
                v_c = nl.load(v_input[bh, off + i_seq_b, i_vdim])
                acc_o = nl.zeros((BLOCK, V_PAD), dtype=nl.float32, buffer=nl.psum)
                scores_acc = nl.zeros((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.psum)

                for f_idx in nl.affine_range(NUM_F_TILES):
                    f_off = f_idx * TILE_F
                    q_f = nl.load(q_input[bh, off + i_seq_b, f_off + i_feat])
                    k_f_t = nl.load_transpose2d(k_input[bh, off + i_seq_b, f_off + i_feat])
                    s_f = nl.load(state[bh, f_off + i_feat_p, i_vdim])
                    acc_o += nl.matmul(q_f, s_f)
                    scores_acc += nl.matmul(q_f, k_f_t)
                    s_f += nl.matmul(k_f_t, v_c)
                    nl.store(state[bh, f_off + i_feat_p, i_vdim], value=s_f)

                scores = nl.copy(scores_acc)
                scores_masked = nl.multiply(scores, nl.load(c_mask))
                acc_o += nl.matmul(scores_masked, v_c)
                nl.store(out[bh, off + i_seq_b, i_vdim], value=nl.copy(acc_o))
        return out

    # Taylor feature map expands D=64 to feat_dim=2145, padded to FEAT_PAD
    q = np.random.randn(BH, SEQ_PAD, FEAT_PAD).astype(np.float32)
    k = np.random.randn(BH, SEQ_PAD, FEAT_PAD).astype(np.float32)
    v = np.random.randn(BH, SEQ_PAD, V_PAD).astype(np.float32)
    print(f"\n=== Chunked kernel: BH={BH}, S={seq_len}(pad={SEQ_PAD}), feat={feat_dim}(pad={FEAT_PAD}), V={value_dim}(pad={V_PAD}) ===")
    kernel(q, k, v)


# ============================================================
# Sliding window kernel
# ============================================================
def bench_sliding(seq_len, dim, num_heads, window_size=64, block_size=64, warmup=10, iters=100):
    BLOCK = block_size
    BH = num_heads
    SEQ_PAD = ((seq_len + BLOCK - 1) // BLOCK) * BLOCK
    D_PAD = ((dim + BLOCK - 1) // BLOCK) * BLOCK
    if D_PAD < 128:
        D_PAD = 128
    NUM_BLOCKS = SEQ_PAD // BLOCK

    if window_size <= 1:
        WINDOW_BLOCKS = 1
    else:
        WINDOW_BLOCKS = (window_size - 2) // BLOCK + 2
    WIN_SEQ = WINDOW_BLOCKS * BLOCK
    assert WIN_SEQ <= 128

    PAD_ROWS = (WINDOW_BLOCKS - 1) * BLOCK
    KV_SEQ_PAD = SEQ_PAD + PAD_ROWS

    _additive_mask = np.full((NUM_BLOCKS, BLOCK, WIN_SEQ), -1e9, dtype=np.float32)
    for i_blk in range(NUM_BLOCKS):
        for r in range(BLOCK):
            t = i_blk * BLOCK + r
            win_start = max(0, t - window_size + 1)
            for w in range(WINDOW_BLOCKS):
                j_orig = i_blk - WINDOW_BLOCKS + 1 + w
                if j_orig < 0:
                    continue
                for c_col in range(BLOCK):
                    s = j_orig * BLOCK + c_col
                    if win_start <= s <= t:
                        _additive_mask[i_blk, r, w * BLOCK + c_col] = 0.0

    i_seq = nl.arange(BLOCK)[:, None]
    i_dim = nl.arange(D_PAD)[None, :]
    i_win = nl.arange(WIN_SEQ)[:, None]
    i_win_col = nl.arange(WIN_SEQ)[None, :]

    @nki.benchmark(warmup=warmup, iters=iters)
    def kernel(q_scaled, k_padded, v_padded, additive_mask):
        out = nl.ndarray((BH, SEQ_PAD, D_PAD), dtype=nl.float32, buffer=nl.hbm, name="out")
        for bh in nl.affine_range(BH):
            for i in nl.affine_range(NUM_BLOCKS):
                q_off = i * BLOCK
                kv_off = i * BLOCK
                q_block = nl.load(q_scaled[bh, q_off + i_seq, i_dim])
                k_win_t = nl.load_transpose2d(k_padded[bh, kv_off + i_win, i_dim])
                v_win = nl.load(v_padded[bh, kv_off + i_win, i_dim])
                scores = nl.copy(nl.matmul(q_block, k_win_t))
                mask = nl.load(additive_mask[i, i_seq, i_win_col])
                masked_scores = nl.add(scores, mask)
                weights = nl.softmax(masked_scores, axis=[1])
                out_block = nl.copy(nl.matmul(weights, v_win))
                nl.store(out[bh, q_off + i_seq, i_dim], value=out_block)
        return out

    q = np.random.randn(BH, SEQ_PAD, D_PAD).astype(np.float32)
    k = np.random.randn(BH, KV_SEQ_PAD, D_PAD).astype(np.float32)
    v = np.random.randn(BH, KV_SEQ_PAD, D_PAD).astype(np.float32)
    mask = _additive_mask
    print(f"\n=== Sliding window kernel: BH={BH}, S={seq_len}(pad={SEQ_PAD}), D={dim}(pad={D_PAD}), W={window_size}, win_blocks={WINDOW_BLOCKS} ===")
    kernel(q, k, v, mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baremetal NKI kernel benchmarks")
    parser.add_argument("--kernel", choices=["parallel", "chunked", "sliding", "all"], default="all")
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--h", type=int, default=2)
    parser.add_argument("--s", type=int, default=256)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--w", type=int, default=64, help="Window size for sliding kernel")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    bh = args.b * args.h
    feat_dim = 2145  # Taylor feature map output dim for d=64

    if args.kernel in ("parallel", "all"):
        bench_parallel(args.s, args.d, bh, warmup=args.warmup, iters=args.iters)
    if args.kernel in ("chunked", "all"):
        bench_chunked(args.s, feat_dim, args.d, bh, warmup=args.warmup, iters=args.iters)
    if args.kernel in ("sliding", "all"):
        bench_sliding(args.s, args.d, bh, window_size=args.w, warmup=args.warmup, iters=args.iters)

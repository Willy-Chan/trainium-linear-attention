"""
NKI kernel for sliding window softmax causal attention on AWS Trainium.

This is the third component of full BASED attention:
  final_output = linear_attention(Q,K,V) + sliding_window_softmax(Q,K,V)

For each token i, attends only to tokens max(0, i-W+1)..i using standard
softmax (not the polynomial approximation). This captures local patterns
that the linear attention component misses.

Design: For each Q-block, loads the full window of K/V as a single
concatenated tile (WINDOW_BLOCKS * BLOCK rows). Uses a precomputed additive
mask to handle causal masking and window boundary validity. The caller
zero-pads K/V at the start so indexing is uniform.

Constraint: WINDOW_BLOCKS * BLOCK must be <= 128 (Trainium matmul
contraction limit). With BLOCK=64, this means window_size <= 128.

Caller pre-scales Q by 1/sqrt(D), pads to (BH, SEQ_PAD, D_PAD).
K/V are zero-padded at the front by (WINDOW_BLOCKS-1)*BLOCK rows.
"""

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

_kernel_cache = {}


def get_sliding_window_kernel(seq_len, dim, num_heads, window_size=64, block_size=64):
    """Return (kernel_fn, SEQ_PAD, D_PAD, KV_SEQ_PAD, additive_mask_np)."""
    key = (seq_len, dim, num_heads, window_size, block_size)
    if key not in _kernel_cache:
        _kernel_cache[key] = _build_kernel(seq_len, dim, num_heads, window_size, block_size)
    return _kernel_cache[key]


def _build_kernel(seq_len, dim, num_heads, window_size, block_size):
    BLOCK = block_size
    BH = num_heads
    SEQ_PAD = ((seq_len + BLOCK - 1) // BLOCK) * BLOCK
    D_PAD = ((dim + BLOCK - 1) // BLOCK) * BLOCK
    if D_PAD < 128:
        D_PAD = 128
    NUM_BLOCKS = SEQ_PAD // BLOCK

    # A window of W tokens can straddle block boundaries. The first token in
    # Q-block i needs tokens from block i-1 even when W == BLOCK. The correct
    # count is: 1 if W <= 1, else floor((W-2)/BLOCK) + 2.
    if window_size <= 1:
        WINDOW_BLOCKS = 1
    else:
        WINDOW_BLOCKS = (window_size - 2) // BLOCK + 2
    WIN_SEQ = WINDOW_BLOCKS * BLOCK
    assert WIN_SEQ <= 128, (
        f"Window {WIN_SEQ} tokens exceeds matmul contraction limit 128. "
        f"Reduce window_size or increase block_size."
    )

    PAD_ROWS = (WINDOW_BLOCKS - 1) * BLOCK
    KV_SEQ_PAD = SEQ_PAD + PAD_ROWS

    # Precompute additive mask: (NUM_BLOCKS, BLOCK, WIN_SEQ)
    # For each (Q-row, K-col) pair, a score is valid (0.0) if:
    #   1. The K-token exists (original block index >= 0)
    #   2. The K-token is within the causal window: s <= t AND s >= t - W + 1
    # Invalid positions get -1e9 so softmax gives them ~zero weight.
    _additive_mask = np.full((NUM_BLOCKS, BLOCK, WIN_SEQ), -1e9, dtype=np.float32)
    for i_blk in range(NUM_BLOCKS):
        for r in range(BLOCK):
            t = i_blk * BLOCK + r  # absolute query token index
            win_start = max(0, t - window_size + 1)
            for w in range(WINDOW_BLOCKS):
                j_orig = i_blk - WINDOW_BLOCKS + 1 + w
                if j_orig < 0:
                    continue  # padding block — stays -inf
                for c in range(BLOCK):
                    s = j_orig * BLOCK + c  # absolute key token index
                    if win_start <= s <= t:
                        _additive_mask[i_blk, r, w * BLOCK + c] = 0.0

    # Tile indices
    i_seq = nl.arange(BLOCK)[:, None]        # (BLOCK, 1) partition dim
    i_dim = nl.arange(D_PAD)[None, :]        # (1, D_PAD) free dim
    i_win = nl.arange(WIN_SEQ)[:, None]      # (WIN_SEQ, 1) partition dim for window K/V
    i_win_col = nl.arange(WIN_SEQ)[None, :]  # (1, WIN_SEQ) free dim for scores/mask

    @nki.jit
    def _kernel(q_scaled, k_padded, v_padded, additive_mask):
        """
        SPMD kernel: each program handles one batch*head slice.
        q_scaled:       (BH, SEQ_PAD, D_PAD)
        k_padded:       (BH, KV_SEQ_PAD, D_PAD) — zero-padded at start
        v_padded:       (BH, KV_SEQ_PAD, D_PAD) — zero-padded at start
        additive_mask:  (NUM_BLOCKS, BLOCK, WIN_SEQ) — causal + validity mask
        Returns: out    (BH, SEQ_PAD, D_PAD)
        """
        out = nl.ndarray((BH, SEQ_PAD, D_PAD), dtype=nl.float32,
                         buffer=nl.shared_hbm, name="out")

        bh = nl.program_id(0)

        for i in nl.affine_range(NUM_BLOCKS):
            q_off = i * BLOCK
            kv_off = i * BLOCK

            q_block = nl.load(q_scaled[bh, q_off + i_seq, i_dim])

            k_win_t = nl.load_transpose2d(
                k_padded[bh, kv_off + i_win, i_dim]
            )

            v_win = nl.load(v_padded[bh, kv_off + i_win, i_dim])

            scores = nl.copy(nl.matmul(q_block, k_win_t))

            mask = nl.load(additive_mask[i, i_seq, i_win_col])
            masked_scores = nl.add(scores, mask)

            weights = nl.softmax(masked_scores, axis=[1])

            out_block = nl.copy(nl.matmul(weights, v_win))

            nl.store(out[bh, q_off + i_seq, i_dim], value=out_block)

        return out

    def kernel(q_scaled, k_padded, v_padded, additive_mask):
        """Wrapper that launches SPMD kernel across BH programs."""
        return _kernel[BH](q_scaled, k_padded, v_padded, additive_mask)

    return kernel, SEQ_PAD, D_PAD, KV_SEQ_PAD, _additive_mask

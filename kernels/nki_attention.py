"""
NKI kernel for fused BASED causal linear attention on AWS Trainium.

Takes batched Q, K, V (all heads flattened into batch dim) and computes causal
attention with 2nd-order Taylor polynomial activation on Q@K.T scores.

Algorithm (per Q-block i, iterating K/V-blocks j = 0..i):
  A[t,s]   = Q_i[t,:] · K_j[s,:]       (BLOCK x BLOCK attention scores)
  poly     = 1 + A + 0.5*A^2            (Taylor polynomial activation)
  masked   = poly * causal_mask          (lower-tri for j==i, all-ones for j<i)
  num_i   += masked @ V_j
  den_i   += row_sum(masked)

Caller pre-scales Q by 1/sqrt(D), pads to (BH, SEQ_PAD, D_PAD).
Kernel returns (num, den); caller computes out = num / (den + eps).
"""

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

_kernel_cache = {}


def get_based_attn_kernel(seq_len, dim, num_heads, block_size=64):
    """Return (kernel_fn, SEQ_PAD, D_PAD), building and caching as needed."""
    key = (seq_len, dim, num_heads, block_size)
    if key not in _kernel_cache:
        _kernel_cache[key] = _build_kernel(seq_len, dim, num_heads, block_size)
    return _kernel_cache[key]


def _build_kernel(seq_len, dim, num_heads, block_size):
    BLOCK = block_size
    BH = num_heads
    SEQ_PAD = ((seq_len + BLOCK - 1) // BLOCK) * BLOCK
    D_PAD = ((dim + BLOCK - 1) // BLOCK) * BLOCK
    if D_PAD < 128:
        D_PAD = 128
    NUM_BLOCKS = SEQ_PAD // BLOCK

    _ones_bb = np.ones((BLOCK, BLOCK), dtype=np.float32)
    _lower_mask = np.tril(_ones_bb)

    # mgrid indices for loading/storing 2D tiles from 3D tensors
    i_seq = nl.arange(BLOCK)[:, None]   # (BLOCK, 1) — partition dim
    i_dim = nl.arange(D_PAD)[None, :]   # (1, D_PAD) — free dim
    i_one = nl.arange(1)[None, :]       # (1, 1)     — free dim for den

    @nki.jit
    def _kernel(q_scaled, k_input, v_input):
        """
        SPMD kernel: each program handles one batch*head slice.
        q_scaled: (BH, SEQ_PAD, D_PAD) — Q pre-scaled by 1/sqrt(D)
        k_input:  (BH, SEQ_PAD, D_PAD)
        v_input:  (BH, SEQ_PAD, D_PAD)
        Returns: (num_out (BH, SEQ_PAD, D_PAD), den_out (BH, SEQ_PAD, 1))
        """
        num_out = nl.ndarray((BH, SEQ_PAD, D_PAD), dtype=nl.float32,
                             buffer=nl.shared_hbm, name="num_out")
        den_out = nl.ndarray((BH, SEQ_PAD, 1), dtype=nl.float32,
                             buffer=nl.shared_hbm, name="den_out")

        c_ones = nl.shared_constant(_ones_bb, dtype=nl.float32)
        c_mask = nl.shared_constant(_lower_mask, dtype=nl.float32)

        bh = nl.program_id(0)

        for i in nl.affine_range(NUM_BLOCKS):
            q_off = i * BLOCK

            q_block = nl.load(q_scaled[bh, q_off + i_seq, i_dim])

            acc_num = nl.zeros((BLOCK, D_PAD), dtype=nl.float32, buffer=nl.psum)
            acc_den = nl.zeros((BLOCK, 1), dtype=nl.float32, buffer=nl.psum)

            # Off-diagonal causal blocks j < i
            for j in nl.affine_range(i):
                k_off = j * BLOCK
                k_t = nl.load_transpose2d(k_input[bh, k_off + i_seq, i_dim])
                v_j = nl.load(v_input[bh, k_off + i_seq, i_dim])

                sc = nl.copy(nl.matmul(q_block, k_t))
                sq = nl.multiply(sc, sc)
                # poly = 1 + sc + 0.5*sc^2: use scalar multiply instead of matrix multiply
                poly = nl.add(nl.add(nl.load(c_ones), sc), nl.multiply(sq, 0.5))

                acc_num += nl.matmul(poly, v_j)
                # Use nl.reduce on scalar engine instead of matmul with ones column
                acc_den += nl.sum(poly, axis=[1], keepdims=True)

            # Diagonal causal block j == i
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

    def kernel(q_scaled, k_input, v_input):
        """Wrapper that launches SPMD kernel across BH programs."""
        return _kernel[BH](q_scaled, k_input, v_input)

    return kernel, SEQ_PAD, D_PAD

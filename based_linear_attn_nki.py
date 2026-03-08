"""
NKI kernel for fused BASED causal linear attention on AWS Trainium.

Takes raw Q, K, V (single head, already projected) and computes causal attention
with 2nd-order Taylor polynomial activation on Q@K.T scores — matching the
TileLang reference kernel (parallel_based_tilelang.py).

Algorithm (per Q-block i, iterating K/V-blocks j = 0..i):
  A[t,s]   = Q_i[t,:] · K_j[s,:]       (BLOCK x BLOCK attention scores)
  poly     = 1 + A + 0.5*A^2            (Taylor polynomial activation)
  masked   = poly * causal_mask          (lower-tri for j==i, all-ones for j<i)
  num_i   += masked @ V_j
  den_i   += row_sum(masked)

Caller pre-scales Q by 1/sqrt(D), pads to (SEQ_PAD, D_PAD).
Kernel returns (num, den); caller computes out = num / (den + eps).
"""

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

_kernel_cache = {}


def get_based_attn_kernel(seq_len, dim, block_size=64):
    """Return (kernel_fn, SEQ_PAD, D_PAD), building and caching as needed."""
    key = (seq_len, dim, block_size)
    if key not in _kernel_cache:
        _kernel_cache[key] = _build_kernel(seq_len, dim, block_size)
    return _kernel_cache[key]


def _build_kernel(seq_len, dim, block_size):
    BLOCK = block_size
    SEQ_PAD = ((seq_len + BLOCK - 1) // BLOCK) * BLOCK
    D_PAD = ((dim + BLOCK - 1) // BLOCK) * BLOCK
    if D_PAD < 128:
        D_PAD = 128
    NUM_BLOCKS = SEQ_PAD // BLOCK

    _ones_bb = np.ones((BLOCK, BLOCK), dtype=np.float32)
    _half_bb = np.full((BLOCK, BLOCK), 0.5, dtype=np.float32)
    _ones_col = np.ones((BLOCK, 1), dtype=np.float32)
    _lower_mask = np.tril(_ones_bb)

    @nki.jit
    def kernel(q_scaled, k_input, v_input):
        """
        q_scaled: (SEQ_PAD, D_PAD) — Q pre-scaled by 1/sqrt(D)
        k_input:  (SEQ_PAD, D_PAD)
        v_input:  (SEQ_PAD, D_PAD)
        Returns: (num_out (SEQ_PAD, D_PAD), den_out (SEQ_PAD, 1))
        """
        num_out = nl.ndarray((SEQ_PAD, D_PAD), dtype=nl.float32,
                             buffer=nl.hbm, name="num_out")
        den_out = nl.ndarray((SEQ_PAD, 1), dtype=nl.float32,
                             buffer=nl.hbm, name="den_out")

        c_ones = nl.shared_constant(_ones_bb, dtype=nl.float32)
        c_half = nl.shared_constant(_half_bb, dtype=nl.float32)
        c_col = nl.shared_constant(_ones_col, dtype=nl.float32)
        c_mask = nl.shared_constant(_lower_mask, dtype=nl.float32)
        for i in nl.affine_range(NUM_BLOCKS):
            q_off = i * BLOCK
            q_block = nl.load(q_scaled[q_off:q_off + BLOCK, :])

            # Allocated ONCE per hardware loop iteration
            acc_num = nl.zeros((BLOCK, D_PAD), dtype=nl.float32, buffer=nl.psum)
            acc_den = nl.zeros((BLOCK, 1), dtype=nl.float32, buffer=nl.psum)

            # Off-diagonal causal blocks j < i
            for j in nl.affine_range(i):
                k_off = j * BLOCK
                k_t = nl.load_transpose2d(k_input[k_off:k_off + BLOCK, :])
                v_j = nl.load(v_input[k_off:k_off + BLOCK, :])

                sc = nl.copy(nl.matmul(q_block, k_t))
                sq = nl.multiply(sc, sc)
                half_sq = nl.multiply(sq, nl.load(c_half))
                poly = nl.multiply(
                    nl.add(nl.add(nl.load(c_ones), sc), half_sq),
                    nl.load(c_ones), # block_mask
                )

                # Use += as NKI expects for native PSUM accumulation
                acc_num += nl.matmul(poly, v_j)
                acc_den += nl.matmul(poly, nl.load(c_col))

            # Diagonal causal block j == i
            k_off = i * BLOCK
            k_t = nl.load_transpose2d(k_input[k_off:k_off + BLOCK, :])
            v_j = nl.load(v_input[k_off:k_off + BLOCK, :])
            
            sc = nl.copy(nl.matmul(q_block, k_t))
            sq = nl.multiply(sc, sc)
            half_sq = nl.multiply(sq, nl.load(c_half))
            poly = nl.multiply(
                nl.add(nl.add(nl.load(c_ones), sc), half_sq),
                nl.load(c_mask),
            )
            
            # Use += as NKI expects for native PSUM accumulation
            acc_num += nl.matmul(poly, v_j)
            acc_den += nl.matmul(poly, nl.load(c_col))

            nl.store(num_out[q_off:q_off + BLOCK, :], value=nl.copy(acc_num))
            nl.store(den_out[q_off:q_off + BLOCK, :], value=nl.copy(acc_den))
        return num_out, den_out

    return kernel, SEQ_PAD, D_PAD

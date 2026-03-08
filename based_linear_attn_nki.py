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
    _zeros_bd = np.zeros((BLOCK, D_PAD), dtype=np.float32)
    _zeros_b1 = np.zeros((BLOCK, 1), dtype=np.float32)

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
        c_z_bd = nl.shared_constant(_zeros_bd, dtype=nl.float32)
        c_z_b1 = nl.shared_constant(_zeros_b1, dtype=nl.float32)

        for i in range(NUM_BLOCKS):
            q_off = i * BLOCK
            # Q loaded normally: (par=BLOCK_t, free=D_PAD) = (64, 128)
            q_block = nl.load(q_scaled[q_off:q_off + BLOCK, :])

            acc_num = nl.load(c_z_bd)
            acc_den = nl.load(c_z_b1)

            for j in range(i + 1):
                k_off = j * BLOCK
                # K loaded transposed: (par=D_PAD, free=BLOCK_s) = (128, 64)
                k_t = nl.load_transpose2d(k_input[k_off:k_off + BLOCK, :])
                # V loaded normally: (par=BLOCK_s, free=D_PAD) = (64, 128)
                v_j = nl.load(v_input[k_off:k_off + BLOCK, :])

                # A[t,s] = Q[t,:] @ K[s,:].T
                # q_block (64, 128) @ k_t (128, 64) -> (64, 64)
                sc = nl.copy(nl.matmul(q_block, k_t))

                # polynomial: phi(x) = 1 + x + 0.5*x^2, with causal mask
                # Lower-tri for diagonal block (j==i), all-ones for j<i.
                block_mask = nl.load(c_mask) if j == i else nl.load(c_ones)
                sq = nl.multiply(sc, sc)
                half_sq = nl.multiply(sq, nl.load(c_half))
                poly = nl.multiply(
                    nl.add(nl.add(nl.load(c_ones), sc), half_sq),
                    block_mask,
                )

                # numerator: poly @ V -> (BLOCK_t, D_PAD) = (64, 128)
                # poly (64, 64) @ v_j (64, 128) -> (64, 128)
                acc_num[...] = nl.add(acc_num,
                                      nl.copy(nl.matmul(poly, v_j)))

                # denominator: poly @ ones_col -> (BLOCK_t, 1) = (64, 1)
                # poly (64, 64) @ ones (64, 1) -> (64, 1)
                acc_den[...] = nl.add(acc_den,
                                      nl.copy(nl.matmul(poly, nl.load(c_col))))

            nl.store(num_out[q_off:q_off + BLOCK, :], value=acc_num)
            nl.store(den_out[q_off:q_off + BLOCK, :], value=acc_den)

        return num_out, den_out

    return kernel, SEQ_PAD, D_PAD

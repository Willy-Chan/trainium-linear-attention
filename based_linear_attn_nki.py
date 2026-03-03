"""
NKI kernel for BASED linear attention (causal) on AWS Trainium.

Reformulated as matrix ops (no sequential loop, no per-timestep partition slicing):
  A[t,s] = sum_f q[t,f]*k[s,f]  (linear attention scores)
  num = lower_tri(A) @ V          (causal masked attention)
  den = row_sum(lower_tri(A))     (normalizer)

All dimensions padded to multiples of 32 for trn1 alignment:
  T_PAD=32 (from T=16), FPAD=64 (from F=45), D_PAD=32 (from D=16).
Caller pads inputs; kernel returns padded outputs (caller extracts [:T, :D]).
"""

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

T, T_PAD, FPAD, D, D_PAD = 16, 32, 64, 16, 32


@nki.jit
def based_linear_attn_nki_kernel(q_input, k_input, v_input):
    """
    q_input: (T_PAD, FPAD), k_input: (T_PAD, FPAD), v_input: (T_PAD, D_PAD).
    All zero-padded by caller.
    Returns (num (T_PAD, D_PAD), den (T_PAD, 1)).
    """
    assert q_input.shape == (T_PAD, FPAD)
    assert k_input.shape == (T_PAD, FPAD)
    assert v_input.shape == (T_PAD, D_PAD)

    # Transposed loads: (FPAD=64 partitions, T_PAD=32 free) — all 32-aligned
    q_t = nl.load_transpose2d(q_input)  # (FPAD, T_PAD)
    k_t = nl.load_transpose2d(k_input)  # (FPAD, T_PAD)

    # Normal load: (T_PAD=32 partitions, D_PAD=32 free)
    v_pad = nl.load(v_input)  # (T_PAD, D_PAD)

    # A_T[s,t] = sum_f k[s,f]*q[t,f] = A[t,s]  (transpose of attention scores)
    # nc_matmul: stationary.T @ moving, contraction along FPAD=64
    A_T_psum = nl.matmul(k_t, q_t, transpose_x=True)  # PSUM (T_PAD, T_PAD)
    A_T = nl.copy(A_T_psum)  # SBUF (32, 32)

    # Upper-triangular causal mask: A_T[s,t] kept when s <= t
    # (equivalent to lower-triangular on A[t,s])
    upper_mask_hbm = nl.shared_constant(
        np.triu(np.ones((T_PAD, T_PAD), dtype=np.float32)), dtype=nl.float32
    )
    upper_mask = nl.load(upper_mask_hbm)
    causal_A_T = nl.multiply(A_T, upper_mask)  # (T_PAD, T_PAD)

    # Numerator: causal_A_T.T @ v_pad = causal_A @ V = (T_PAD, D_PAD)
    num_psum = nl.matmul(causal_A_T, v_pad, transpose_x=True)

    # Denominator: causal_A_T.T @ ones = row_sum(causal_A) = (T_PAD, 1)
    ones_col_hbm = nl.shared_constant(
        np.ones((T_PAD, 1), dtype=np.float32), dtype=nl.float32
    )
    ones_col = nl.load(ones_col_hbm)
    den_psum = nl.matmul(causal_A_T, ones_col, transpose_x=True)

    # Store outputs to HBM
    num_out = nl.ndarray((T_PAD, D_PAD), dtype=nl.float32, buffer=nl.hbm, name="num_out")
    den_out = nl.ndarray((T_PAD, 1), dtype=nl.float32, buffer=nl.hbm, name="den_out")
    nl.store(num_out, value=nl.copy(num_psum))
    nl.store(den_out, value=nl.copy(den_psum))

    return num_out, den_out

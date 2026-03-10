"""
NKI kernel for chunked/recurrent linear attention on AWS Trainium.

Uses explicit feature-mapped Q/K (computed outside the kernel) with an HBM-resident
recurrent state S of shape (FEAT_PAD, V_PAD). Each chunk:
  1. Inter-chunk: out += Q_chunk @ S
  2. Intra-chunk: out += causal_mask(Q_chunk @ K_chunk^T) @ V_chunk
  3. State update: S += K_chunk^T @ V_chunk

The caller applies the Taylor feature map (phi(x) = [1, x, 0.5*x^2]) before calling.
"""

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

_kernel_cache = {}


def get_chunked_attn_kernel(seq_len, feat_dim, value_dim, num_heads, chunk_size=64):
    key = (seq_len, feat_dim, value_dim, num_heads, chunk_size)
    if key not in _kernel_cache:
        _kernel_cache[key] = _build_chunked_kernel(seq_len, feat_dim, value_dim, num_heads, chunk_size)
    return _kernel_cache[key]


def _build_chunked_kernel(seq_len, feat_dim, value_dim, num_heads, chunk_size):
    BLOCK = chunk_size
    BH = num_heads
    TILE_F = 128  # contraction tile; must be <= 128 for TE partition limits

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

    # mgrid indices for 3D tensor access
    i_seq_b = nl.arange(BLOCK)[:, None]    # (BLOCK, 1) — partition dim for seq tiles
    i_feat_p = nl.arange(TILE_F)[:, None]  # (TILE_F, 1) — partition dim for feature tiles
    i_feat = nl.arange(TILE_F)[None, :]    # (1, TILE_F) — free dim
    i_vdim = nl.arange(V_PAD)[None, :]     # (1, V_PAD) — free dim

    @nki.jit
    def kernel(q_input, k_input, v_input):
        """
        q_input: (BH, SEQ_PAD, FEAT_PAD)
        k_input: (BH, SEQ_PAD, FEAT_PAD)
        v_input: (BH, SEQ_PAD, V_PAD)
        Returns out: (BH, SEQ_PAD, V_PAD) numerator-only output
        """
        out = nl.ndarray((BH, SEQ_PAD, V_PAD), dtype=nl.float32, buffer=nl.hbm, name="out")
        # Each head gets its own state region
        state = nl.ndarray((BH, FEAT_PAD, V_PAD), dtype=nl.float32, buffer=nl.hbm, name="state")

        c_mask = nl.shared_constant(_lower_mask, dtype=nl.float32)
        c_zero_fv = nl.shared_constant(_zeros_fv, dtype=nl.float32)

        for bh in nl.affine_range(BH):
            # Initialize recurrent state for this head to zeros
            for f_idx in nl.affine_range(NUM_F_TILES):
                f_off = f_idx * TILE_F
                nl.store(state[bh, f_off + i_feat_p, i_vdim], value=nl.load(c_zero_fv))

            for c in nl.sequential_range(NUM_CHUNKS):
                off = c * BLOCK

                # Load V chunk once
                v_c = nl.load(v_input[bh, off + i_seq_b, i_vdim])  # (BLOCK, V_PAD)

                # Inter-chunk output accumulator
                acc_o = nl.zeros((BLOCK, V_PAD), dtype=nl.float32, buffer=nl.psum)

                # Intra-chunk score accumulator
                scores_acc = nl.zeros((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.psum)

                # Fused loop: compute Q@S, Q@K^T, and update S += K^T@V
                # in a single pass over feature tiles (caches K^T in SBUF)
                for f_idx in nl.affine_range(NUM_F_TILES):
                    f_off = f_idx * TILE_F

                    q_f = nl.load(q_input[bh, off + i_seq_b, f_off + i_feat])  # (BLOCK, 128)
                    k_f_t = nl.load_transpose2d(
                        k_input[bh, off + i_seq_b, f_off + i_feat]
                    )  # (128, BLOCK)
                    s_f = nl.load(state[bh, f_off + i_feat_p, i_vdim])  # (128, V_PAD)

                    # Inter-chunk: Q_f @ S_f (read state before update)
                    acc_o += nl.matmul(q_f, s_f)

                    # Intra-chunk scores: Q_f @ K_f^T
                    scores_acc += nl.matmul(q_f, k_f_t)

                    # Update recurrent state: S_f += K_f^T @ V (reuses cached k_f_t)
                    s_f += nl.matmul(k_f_t, v_c)
                    nl.store(state[bh, f_off + i_feat_p, i_vdim], value=s_f)

                # Intra-chunk: mask(scores) @ V
                scores = nl.copy(scores_acc)
                scores_masked = nl.multiply(scores, nl.load(c_mask))
                acc_o += nl.matmul(scores_masked, v_c)

                nl.store(out[bh, off + i_seq_b, i_vdim], value=nl.copy(acc_o))

        return out

    return kernel, SEQ_PAD, FEAT_PAD, V_PAD

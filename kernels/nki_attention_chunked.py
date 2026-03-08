# """
# NKI kernel for chunked/recurrent BASED causal linear attention.
# Implements the 2nd-order Taylor polynomial feature map expansion.
# """

# import numpy as np
# import neuronxcc.nki as nki
# import neuronxcc.nki.language as nl

# _kernel_cache = {}


# def get_chunked_based_kernel(seq_len, dim, chunk_size=64):
#     """Return (kernel_fn, SEQ_PAD, D_PAD), building and caching as needed."""
#     key = (seq_len, dim, chunk_size)
#     if key not in _kernel_cache:
#         _kernel_cache[key] = _build_chunked_based_kernel(seq_len, dim, chunk_size)
#     return _kernel_cache[key]


# def _build_chunked_based_kernel(seq_len, dim, chunk_size):
#     BLOCK = chunk_size
#     SEQ_PAD = ((seq_len + BLOCK - 1) // BLOCK) * BLOCK
    
#     # We strictly bound D_PAD. If D > 64, the D^2 state expands to >8MB 
#     # and will crash the compiler during the unrolling phase.
#     D_PAD = ((dim + BLOCK - 1) // BLOCK) * BLOCK
#     if D_PAD < 64: 
#         D_PAD = 64
#     assert D_PAD <= 64, "D_PAD > 64 will cause PSUM exhaustion in this fully unrolled implementation."

#     NUM_CHUNKS = SEQ_PAD // BLOCK

#     # Pre-compute constants for NKI
#     _ones_bb = np.ones((BLOCK, BLOCK), dtype=np.float32)
#     _half_bb = np.full((BLOCK, BLOCK), 0.5, dtype=np.float32)
#     _lower_mask = np.tril(_ones_bb)
    
#     _ones_col = np.ones((BLOCK, 1), dtype=np.float32)
#     _ones_row = np.ones((1, BLOCK), dtype=np.float32)

#     @nki.jit
#     def kernel(q_scaled, k_input, v_input):
#         out = nl.ndarray((SEQ_PAD, D_PAD), dtype=nl.float32, buffer=nl.hbm, name="out")
        
#         c_mask = nl.shared_constant(_lower_mask, dtype=nl.float32)
#         c_half = nl.shared_constant(_half_bb, dtype=nl.float32)
#         c_ones = nl.shared_constant(_ones_bb, dtype=nl.float32)
#         c_ones_col = nl.shared_constant(_ones_col, dtype=nl.float32)
#         c_ones_row = nl.shared_constant(_ones_row, dtype=nl.float32)

#         # ------------------------------------------------------------------
#         # FRACTURED PERSISTENT STATE IN PSUM
#         # ------------------------------------------------------------------
#         # S0 handles the +1 term
#         S0_state = nl.zeros((1, D_PAD), dtype=nl.float32, buffer=nl.psum)
        
#         # S1 handles the linear (x) term
#         S1_state = nl.zeros((D_PAD, D_PAD), dtype=nl.float32, buffer=nl.psum)
        
#         # S2 handles the quadratic (0.5 * x^2) term.
#         # We represent D^2 x D as a Python list of D tensors of shape (D, D).
#         # The compiler will unroll this into separate hardware PSUM registers.
#         S2_state = [
#             nl.zeros((D_PAD, D_PAD), dtype=nl.float32, buffer=nl.psum) 
#             for _ in range(D_PAD)
#         ]

#         for c in nl.affine_range(NUM_CHUNKS):
#             off = c * BLOCK

#             # 1. Load chunk inputs
#             q_c = nl.load(q_scaled[off : off + BLOCK, :])
#             k_c = nl.load(k_input[off : off + BLOCK, :]) 
#             k_c_t = nl.load_transpose2d(k_input[off : off + BLOCK, :])
#             v_c = nl.load(v_input[off : off + BLOCK, :])

#             # ------------------------------------------------------------------
#             # 2. Inter-chunk contribution
#             # ------------------------------------------------------------------
#             # Start with the constant term: Broadcast S0 (1, D) to (BLOCK, D)
#             acc_o = nl.matmul(nl.load(c_ones_col), nl.copy(S0_state))
            
#             # Add linear term: Q @ S1
#             acc_o += nl.matmul(q_c, nl.copy(S1_state))
            
#             # Add quadratic term: 0.5 * (Q \otimes Q) @ S2
#             # We compute this by tiling over the D_PAD dimension to avoid SBUF limits
#             for d in range(D_PAD):
#                 # Slice the d-th column of Q and broadcast-multiply with Q
#                 q_col = q_c[:, d:d+1] 
#                 q_tile = nl.multiply(q_col, q_c) 
                
#                 # Matmul with the corresponding tile of S2
#                 quad_tile = nl.matmul(q_tile, nl.copy(S2_state[d]))
#                 acc_o += nl.multiply(quad_tile, 0.5)

#             # ------------------------------------------------------------------
#             # 3. Intra-chunk contribution (Standard Taylor Expansion)
#             # ------------------------------------------------------------------
#             scores = nl.matmul(q_c, k_c_t)
#             sq_scores = nl.multiply(scores, scores)
#             half_sq = nl.multiply(sq_scores, nl.load(c_half))
            
#             poly = nl.add(nl.add(nl.load(c_ones), scores), half_sq)
#             poly_masked = nl.multiply(poly, nl.load(c_mask))
            
#             # Accumulate intra into inter
#             acc_o += nl.matmul(poly_masked, v_c)

#             # Store final chunk output to HBM
#             nl.store(out[off : off + BLOCK, :], value=nl.copy(acc_o))

#             # ------------------------------------------------------------------
#             # 4. Update recurrent state
#             # ------------------------------------------------------------------
#             # S0 += \sum V (Summing rows of V)
#             S0_state += nl.matmul(nl.load(c_ones_row), v_c)
            
#             # S1 += K^T @ V
#             S1_state += nl.matmul(k_c_t, v_c)
            
#             # S2 += (K \otimes K)^T @ V
#             # Tiled to match the S2_state layout
#             for d in range(D_PAD):
#                 k_col = k_c[:, d:d+1]
#                 k_tile = nl.multiply(k_col, k_c)
#                 # Accumulate directly into the specific PSUM tile
#                 S2_state[d] += nl.matmul(k_tile, v_c, transpose_x=True)

#         return out

#     return kernel, SEQ_PAD, D_PAD

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

_kernel_cache = {}


def get_chunked_attn_kernel(seq_len, feat_dim, value_dim, chunk_size=64):
    key = (seq_len, feat_dim, value_dim, chunk_size)
    if key not in _kernel_cache:
        _kernel_cache[key] = _build_chunked_kernel(seq_len, feat_dim, value_dim, chunk_size)
    return _kernel_cache[key]


def _build_chunked_kernel(seq_len, feat_dim, value_dim, chunk_size):
    BLOCK = chunk_size
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

    @nki.jit
    def kernel(q_input, k_input, v_input):
        """
        q_input: (SEQ_PAD, FEAT_PAD)
        k_input: (SEQ_PAD, FEAT_PAD)
        v_input: (SEQ_PAD, V_PAD)
        Returns out: (SEQ_PAD, V_PAD) numerator-only output
        """
        out = nl.ndarray((SEQ_PAD, V_PAD), dtype=nl.float32, buffer=nl.hbm, name="out")
        state = nl.ndarray((FEAT_PAD, V_PAD), dtype=nl.float32, buffer=nl.hbm, name="state")

        c_mask = nl.shared_constant(_lower_mask, dtype=nl.float32)
        c_zero_fv = nl.shared_constant(_zeros_fv, dtype=nl.float32)

        # Initialize recurrent state in HBM to zeros, tile by tile.
        for f_idx in nl.affine_range(NUM_F_TILES):
            f_off = f_idx * TILE_F
            nl.store(state[f_off : f_off + TILE_F, :], value=nl.load(c_zero_fv))

        for c in nl.affine_range(NUM_CHUNKS):
            off = c * BLOCK

            # Load V chunk once; reused for both intra and state update.
            v_c = nl.load(v_input[off : off + BLOCK, :])  # (BLOCK, V_PAD)

            # Inter-chunk output accumulator for this chunk.
            acc_o = nl.zeros((BLOCK, V_PAD), dtype=nl.float32, buffer=nl.psum)

            # Intra-chunk score accumulator.
            scores_acc = nl.zeros((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.psum)

            # Tile over feature dimension so contraction never exceeds 128.
            for f_idx in nl.affine_range(NUM_F_TILES):
                f_off = f_idx * TILE_F

                q_f = nl.load(q_input[off : off + BLOCK, f_off : f_off + TILE_F])  # (BLOCK, 128)
                k_f_t = nl.load_transpose2d(
                    k_input[off : off + BLOCK, f_off : f_off + TILE_F]
                )  # (128, BLOCK)
                s_f = nl.load(state[f_off : f_off + TILE_F, :])  # (128, V_PAD)

                # Inter-chunk: Q_f @ S_f
                acc_o += nl.matmul(q_f, s_f)

                # Intra-chunk scores: Q_f @ K_f^T
                scores_acc += nl.matmul(q_f, k_f_t)

            # Intra-chunk: mask(scores) @ V
            scores = nl.copy(scores_acc)
            scores_masked = nl.multiply(scores, nl.load(c_mask))
            acc_o += nl.matmul(scores_masked, v_c)

            nl.store(out[off : off + BLOCK, :], value=nl.copy(acc_o))

            # Update recurrent state: S += K^T @ V, tiled by feature.
            for f_idx in nl.affine_range(NUM_F_TILES):
                f_off = f_idx * TILE_F
                k_f_t = nl.load_transpose2d(
                    k_input[off : off + BLOCK, f_off : f_off + TILE_F]
                )  # (128, BLOCK)
                s_f = nl.load(state[f_off : f_off + TILE_F, :])  # (128, V_PAD)
                s_f += nl.matmul(k_f_t, v_c)
                nl.store(state[f_off : f_off + TILE_F, :], value=s_f)

        return out

    return kernel, SEQ_PAD, FEAT_PAD, V_PAD

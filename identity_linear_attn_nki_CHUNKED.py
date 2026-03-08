import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

_kernel_cache = {}


def get_chunked_attn_kernel(seq_len, dim, chunk_size=64):
    key = (seq_len, dim, chunk_size)
    if key not in _kernel_cache:
        _kernel_cache[key] = _build_chunked_kernel(seq_len, dim, chunk_size)
    return _kernel_cache[key]


def _build_chunked_kernel(seq_len, dim, chunk_size):
    BLOCK = chunk_size
    SEQ_PAD = ((seq_len + BLOCK - 1) // BLOCK) * BLOCK
    D_PAD = ((dim + BLOCK - 1) // BLOCK) * BLOCK
    if D_PAD < 128:
        D_PAD = 128
    NUM_CHUNKS = SEQ_PAD // BLOCK

    _ones_bb = np.ones((BLOCK, BLOCK), dtype=np.float32)
    _lower_mask = np.tril(_ones_bb)

    @nki.jit
    def kernel(q_scaled, k_input, v_input):
        """
        q_scaled, k_input, v_input: (SEQ_PAD, D_PAD)
        Returns: out (SEQ_PAD, D_PAD) - raw numerator before normalization
        """
        out = nl.ndarray((SEQ_PAD, D_PAD), dtype=nl.float32, buffer=nl.hbm, name="out")
        c_mask = nl.shared_constant(_lower_mask, dtype=nl.float32)

        # Persistent recurrent state in PSUM
        S_state = nl.zeros((D_PAD, D_PAD), dtype=nl.float32, buffer=nl.psum)

        for c in nl.affine_range(NUM_CHUNKS):
            off = c * BLOCK

            # 1) Load chunk inputs
            q_c = nl.load(q_scaled[off : off + BLOCK, :])
            k_c_t = nl.load_transpose2d(k_input[off : off + BLOCK, :])
            v_c = nl.load(v_input[off : off + BLOCK, :])

            # 2) Inter-chunk contribution: O_inter = Q @ S_prev
            S_sbuf = nl.copy(S_state, dtype=nl.float32)
            acc_o = nl.matmul(q_c, S_sbuf)

            # 3) Intra-chunk contribution: O_intra = Mask(Q @ K.T) @ V
            scores = nl.matmul(q_c, k_c_t)
            scores_masked = nl.multiply(scores, nl.load(c_mask))
            acc_o += nl.matmul(scores_masked, v_c)

            # Store chunk output
            nl.store(out[off : off + BLOCK, :], value=nl.copy(acc_o))

            # 4) Update recurrent state: S_state += K.T @ V
            S_state += nl.matmul(k_c_t, v_c)

        return out

    return kernel, SEQ_PAD, D_PAD

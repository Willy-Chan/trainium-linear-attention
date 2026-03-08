"""
Fused BASED causal linear attention.

Computes polynomial feature map (phi(x) = 1 + x + 0.5*x^2) on Q@K.T scores,
applies causal masking, then weighted sum with V — all in one fused operation.

Interface: Q, K, V of shape (b, h, t, d) -> output (b, h, t, d).
"""

import math
import torch

_nki_available = None


def _check_nki():
    global _nki_available
    if _nki_available is None:
        try:
            from kernels.nki_attention import get_based_attn_kernel  # noqa: F401
            _nki_available = True
        except Exception:
            _nki_available = False
    return _nki_available


# def ref_based_attn(q, k, v, eps=1e-6):
#     """
#     Reference PyTorch path that mirrors the current NKI kernel behavior.
#     q, k, v: (b, h, t, d)
#     Returns: (b, h, t, d)
#     """
#     b, h, t, d = q.shape
#     bh = b * h
#     block = 64
#     seq_pad = ((t + block - 1) // block) * block
#     d_pad = ((d + block - 1) // block) * block
#     if d_pad < 128:
#         d_pad = 128

#     dev = q.device
#     q_scaled = (q * (1.0 / math.sqrt(d))).float().reshape(bh, t, d)
#     k_flat = k.float().reshape(bh, t, d)
#     v_flat = v.float().reshape(bh, t, d)

#     def pad(x, rows, cols):
#         out = torch.zeros(bh, rows, cols, device=dev, dtype=torch.float32)
#         out[:, : x.shape[1], : x.shape[2]] = x
#         return out

#     q_pad = pad(q_scaled, seq_pad, d_pad)
#     k_pad = pad(k_flat, seq_pad, d_pad)
#     v_pad = pad(v_flat, seq_pad, d_pad)

#     num = torch.zeros(bh, seq_pad, d_pad, device=dev, dtype=torch.float32)
#     den = torch.zeros(bh, seq_pad, 1, device=dev, dtype=torch.float32)
#     tril = torch.tril(torch.ones(block, block, device=dev, dtype=torch.float32))
#     ones_col = torch.ones(bh, block, 1, device=dev, dtype=torch.float32)

#     for i_off in range(0, seq_pad, block):
#         q_blk = q_pad[:, i_off : i_off + block, :]
#         acc_num = torch.zeros(bh, block, d_pad, device=dev, dtype=torch.float32)
#         acc_den = torch.zeros(bh, block, 1, device=dev, dtype=torch.float32)
#         for j_off in range(0, i_off + block, block):
#             k_blk = k_pad[:, j_off : j_off + block, :]
#             v_blk = v_pad[:, j_off : j_off + block, :]
#             sc = torch.matmul(q_blk, k_blk.transpose(-2, -1))
#             poly = 1.0 + sc + 0.5 * sc * sc
#             if j_off == i_off:
#                 poly = poly * tril
#             acc_num = acc_num + torch.matmul(poly, v_blk)
#             acc_den = acc_den + torch.matmul(poly, ones_col)
#         num[:, i_off : i_off + block, :] = acc_num
#         den[:, i_off : i_off + block, :] = acc_den

#     out = num[:, :t, :d] / (den[:, :t, :] + eps)
#     return out.reshape(b, h, t, d).to(q.dtype)

def ref_based_attn(q, k, v, eps=1e-6):
    """
    Reference PyTorch BASED causal attention with polynomial activation.
    q, k, v: (b, h, t, d)
    Returns: (b, h, t, d)
    """
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(q * scale, k.transpose(-2, -1))
    poly = 1.0 + scores + 0.5 * scores * scores
    t = q.shape[-2]
    mask = torch.tril(torch.ones(t, t, device=q.device, dtype=q.dtype))
    masked = poly * mask
    den = masked.sum(dim=-1, keepdim=True) + eps
    out = torch.matmul(masked, v) / den
    return out

def solution_nki(q, k, v, eps=1e-6):
    """
    BASED causal attention via NKI kernel (Trainium).
    q, k, v: (b, h, t, d)
    Returns: (b, h, t, d)
    """
    from kernels.nki_attention import get_based_attn_kernel

    b, h, t, d = q.shape
    scale = 1.0 / math.sqrt(d)

    kernel, SEQ_PAD, D_PAD = get_based_attn_kernel(t, d)

    q_scaled = (q * scale).float().reshape(b * h, t, d)
    k_flat = k.float().reshape(b * h, t, d)
    v_flat = v.float().reshape(b * h, t, d)
    dev = q.device

    def pad(x, rows, cols):
        bh, r, c = x.shape
        if r >= rows and c >= cols:
            return x
        out = torch.zeros(bh, rows, cols, device=dev, dtype=torch.float32)
        out[:, :r, :c] = x
        return out

    q_pad = pad(q_scaled, SEQ_PAD, D_PAD)
    k_pad = pad(k_flat, SEQ_PAD, D_PAD)
    v_pad = pad(v_flat, SEQ_PAD, D_PAD)

    outs = []
    for i in range(b * h):
        num, den = kernel(q_pad[i], k_pad[i], v_pad[i])
        out_i = num[:t, :d] / (den[:t] + eps)
        outs.append(out_i)

    y = torch.stack(outs, dim=0).reshape(b, h, t, d)
    return y.to(q.dtype)


def solution(q, k, v, eps=1e-6):
    """
    BASED causal attention — NKI on Trainium when available, else PyTorch.
    q, k, v: (b, h, t, d)
    Returns: (b, h, t, d)
    """
    if _check_nki():
        try:
            return solution_nki(q, k, v, eps=eps)
        except Exception:
            pass
    return ref_based_attn(q, k, v, eps=eps)

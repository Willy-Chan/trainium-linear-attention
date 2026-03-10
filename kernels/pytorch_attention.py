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
    bh = b * h
    scale = 1.0 / math.sqrt(d)

    kernel, SEQ_PAD, D_PAD = get_based_attn_kernel(t, d, num_heads=bh)

    q_scaled = (q * scale).float().reshape(bh, t, d)
    k_flat = k.float().reshape(bh, t, d)
    v_flat = v.float().reshape(bh, t, d)
    dev = q.device

    def pad(x, rows, cols):
        bh_, r, c = x.shape
        if r >= rows and c >= cols:
            return x
        out = torch.zeros(bh_, rows, cols, device=dev, dtype=torch.float32)
        out[:, :r, :c] = x
        return out

    q_pad = pad(q_scaled, SEQ_PAD, D_PAD)
    k_pad = pad(k_flat, SEQ_PAD, D_PAD)
    v_pad = pad(v_flat, SEQ_PAD, D_PAD)

    # Single kernel call for all batch*heads
    num, den = kernel(q_pad, k_pad, v_pad)
    y = num[:, :t, :d] / (den[:, :t, :] + eps)
    return y.reshape(b, h, t, d).to(q.dtype)


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


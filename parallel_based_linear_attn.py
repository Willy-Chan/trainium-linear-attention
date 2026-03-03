"""Fundamental linear attention kernel on QKV for BASED-style attention."""

import torch

# NKI kernel for Trainium (optional)
_NKI_KERNEL = None

def _get_nki_kernel():
    global _NKI_KERNEL
    if _NKI_KERNEL is None:
        try:
            from based_linear_attn_nki import based_linear_attn_nki_kernel
            _NKI_KERNEL = based_linear_attn_nki_kernel
        except Exception as e:
            _NKI_KERNEL = False
            import warnings
            warnings.warn(f"NKI kernel import failed: {e}", UserWarning)
    return _NKI_KERNEL if _NKI_KERNEL is not False else None


def solution_nki(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    BASED causal linear attention via NKI kernel (Trainium).
    q, k: (b, h, t, 1, feat) -> (b*h, t, feat); v: (b, h, t, d, 1) -> (b*h, t, d).
    All dims zero-padded to multiples of 32 for trn1 alignment:
      T->T_PAD=32, feat->FPAD=64, D->D_PAD=32.
    """
    T_PAD, FPAD, D_PAD = 32, 64, 32
    kernel = _get_nki_kernel()
    if kernel is None:
        raise RuntimeError("NKI kernel not available")
    b, h, t, _, feat = q.shape
    _, _, _, d, _ = v.shape

    q_flat = q.squeeze(3).float().reshape(b * h, t, feat)
    k_flat = k.squeeze(3).float().reshape(b * h, t, feat)
    v_flat = v.squeeze(-1).float().reshape(b * h, t, d)

    dev = q_flat.device
    dtype = q_flat.dtype

    # Pad to (T_PAD, FPAD) and (T_PAD, D_PAD)
    def pad2d(x, target_rows, target_cols):
        bh, r, c = x.shape
        if r < target_rows or c < target_cols:
            out = torch.zeros(bh, target_rows, target_cols, device=dev, dtype=dtype)
            out[:, :r, :c] = x
            return out
        return x

    q_pad = pad2d(q_flat, T_PAD, FPAD)
    k_pad = pad2d(k_flat, T_PAD, FPAD)
    v_pad = pad2d(v_flat, T_PAD, D_PAD)

    outs = []
    for i in range(b * h):
        num, den = kernel(q_pad[i], k_pad[i], v_pad[i])
        out_i = num / (den + eps)
        outs.append(out_i[:t, :d])  # extract unpadded region
    y = torch.stack(outs, dim=0).reshape(b, h, t, d)
    return y.to(q.dtype)


def solution(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Linear attention core: (Q, K, V) -> y.

    Expects q, k, v already feature-mapped and shaped as:
      q, k: (b, h, t, 1, feat)
      v:    (b, h, t, d, 1)

    Returns y of shape (b, h, t, d).
    On Trainium with NKI available and shape (1, 2, 16, 1, 45) / (1, 2, 16, 16, 1), uses NKI kernel.
    """
    kernel = _get_nki_kernel()
    use_nki = (
        kernel is not None
        and causal
        and q.dim() == 5
        and q.shape[2] == 16 and q.shape[3] == 1
        and v.shape[2] == 16 and v.shape[3] == 16 and v.shape[4] == 1
    )
    if use_nki:
        return solution_nki(q, k, v, eps=eps)

    # PyTorch fallback
    if causal:
        y = (q * (k * v).cumsum(2)).sum(-1) / ((q * k.cumsum(2)).sum(-1) + eps)
    else:
        y = (q * (k * v).sum(2, True)).sum(-1) / ((q * k.sum(2, True)).sum(-1) + eps)
    return y



import math
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm

from parallel_based_linear_attn import solution

device = xm.xla_device()


def flatten_diag_outer_product_off1(x, y):
    """Compute upper triangle (excl diagonal) and diagonal of outer product. XLA compatible."""
    z = torch.einsum("...i,...j->...ij", x, y)
    N = z.size(-1)
    triu_i, triu_j = torch.triu_indices(N, N, 1)
    diag_idx = torch.arange(N, device=z.device, dtype=torch.long)
    x2_1 = z[..., triu_i, triu_j]
    x2_2 = z[..., diag_idx, diag_idx]
    return x2_1, x2_2


class TaylorFeatureMap(nn.Module):
    """Taylor series feature map for BASED linear attention. Approximates exp(qk^T)."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.r2 = math.sqrt(2)
        self.rd = math.sqrt(head_dim)
        self.rrd = math.sqrt(self.rd)

    def forward(self, x: torch.Tensor):
        x2_1, x2_2 = flatten_diag_outer_product_off1(x, x)
        return torch.cat(
            [
                torch.ones_like(x[..., 0:1]),
                x / self.rrd,
                x2_2 / (self.rd * self.r2),
                x2_1 / self.rd,
            ],
            dim=-1,
        )


class TrainiumBasedLinearAttention(nn.Module):
    """
    BASED parallel linear attention for Trainium.
    Implements the reference forward pass exactly - feature map + normalized linear attention.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        feature_dim: int = 16,
        eps: float = 1e-12,
        causal: bool = True,
    ):
        super().__init__()
        self.hidden_size = dim
        self.num_heads = heads
        self.num_key_value_heads = heads
        self.feature_dim = feature_dim
        self.head_dim = dim // heads
        self.causal = causal
        self.eps = eps

        assert dim % heads == 0

        self.q_proj = nn.Linear(dim, feature_dim * heads, bias=False)
        self.k_proj = nn.Linear(dim, feature_dim * heads, bias=False)
        self.v_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(heads * self.head_dim, dim, bias=False)
        self.dropout = nn.Dropout(0.0)
        self.feature_map = TaylorFeatureMap(feature_dim)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        """
        BASED parallel linear attention - reference implementation.
        hidden_states: (b, t, d)
        """
        b, t, _ = hidden_states.size()
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        q = q.view(b, t, self.num_heads, self.feature_dim).transpose(1, 2)
        k = k.view(b, t, self.num_key_value_heads, self.feature_dim).transpose(1, 2)
        v = v.view(b, t, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Linear attention: apply feature map to q, k
        q, k = self.feature_map(q), self.feature_map(k)
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)

        # Compute attention
        if self.causal:
            y = (q * (k * v).cumsum(2)).sum(-1) / ((q * k.cumsum(2)).sum(-1) + self.eps)
        else:
            y = (q * (k * v).sum(2, True)).sum(-1) / ((q * k.sum(2, True)).sum(-1) + self.eps)

        # (b, h, t, d) -> (b, t, h*d)
        y = y.transpose(1, 2).reshape(b, t, -1)
        y = self.o_proj(y.to(hidden_states.dtype))
        y = self.dropout(y)
        return y.to(hidden_states.dtype)



class TrainiumBasedLinearAttentionSOLUTION(nn.Module):
    """
    BASED parallel linear attention for Trainium.
    Implements the reference forward pass exactly - feature map + normalized linear attention.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        feature_dim: int = 16,
        eps: float = 1e-12,
        causal: bool = True,
    ):
        super().__init__()
        self.hidden_size = dim
        self.num_heads = heads
        self.num_key_value_heads = heads
        self.feature_dim = feature_dim
        self.head_dim = dim // heads
        self.causal = causal
        self.eps = eps

        assert dim % heads == 0

        self.q_proj = nn.Linear(dim, feature_dim * heads, bias=False)
        self.k_proj = nn.Linear(dim, feature_dim * heads, bias=False)
        self.v_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(heads * self.head_dim, dim, bias=False)
        self.dropout = nn.Dropout(0.0)
        self.feature_map = TaylorFeatureMap(feature_dim)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        """
        BASED parallel linear attention - reference implementation.
        hidden_states: (b, t, d)
        """
        b, t, _ = hidden_states.size()
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        q = q.view(b, t, self.num_heads, self.feature_dim).transpose(1, 2)
        k = k.view(b, t, self.num_key_value_heads, self.feature_dim).transpose(1, 2)
        v = v.view(b, t, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Linear attention: apply feature map to q, k
        q, k = self.feature_map(q), self.feature_map(k)
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)

        y = solution(q, k, v, causal=self.causal, eps=self.eps)

        # (b, h, t, d) -> (b, t, h*d)
        y = y.transpose(1, 2).reshape(b, t, -1)
        y = self.o_proj(y.to(hidden_states.dtype))
        y = self.dropout(y)
        return y.to(hidden_states.dtype)
        


# --- Compare only the linear attention kernel (reference vs solution()) ---
if __name__ == "__main__":
    dim, heads, feature_dim = 32, 2, 8
    batch, seq_len = 1, 16
    causal, eps = True, 1e-12

    # One model only to get q,k,v (projections + feature map). We ignore o_proj for the check.
    model = TrainiumBasedLinearAttention(
        dim=dim, heads=heads, feature_dim=feature_dim, causal=causal, eps=eps
    ).to(device)

    x = torch.randn(batch, seq_len, dim).to(device)
    b, t, _ = x.size()
    q, k, v = model.q_proj(x), model.k_proj(x), model.v_proj(x)
    q = q.view(b, t, model.num_heads, model.feature_dim).transpose(1, 2)
    k = k.view(b, t, model.num_key_value_heads, model.feature_dim).transpose(1, 2)
    v = v.view(b, t, model.num_key_value_heads, model.head_dim).transpose(1, 2)
    q, k = model.feature_map(q), model.feature_map(k)
    q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)

    # Reference linear attention kernel (inline)
    if causal:
        y_ref = (q * (k * v).cumsum(2)).sum(-1) / ((q * k.cumsum(2)).sum(-1) + eps)
    else:
        y_ref = (q * (k * v).sum(2, True)).sum(-1) / ((q * k.sum(2, True)).sum(-1) + eps)

    # solution() linear attention kernel
    y_sol = solution(q, k, v, causal=causal, eps=eps)

    torch_xla.sync()

    # --- CORRECTNESS: compare only kernel outputs (b, h, t, d) ---
    ok = torch.allclose(y_ref, y_sol, rtol=1e-5, atol=1e-5)
    max_diff = (y_ref - y_sol).abs().max().item()
    sep = "=" * 60
    print(f"Kernel output shapes: y_ref {y_ref.shape}, y_sol {y_sol.shape}")
    if ok:
        print(f"\n{sep}")
        print("  ***  CORRECT: reference and solution() kernels match  ***")
        print(f"  max |y_ref - y_sol| = {max_diff:.2e}")
        print(sep)
    else:
        print(f"\n{sep}")
        print("  ***  FAIL: reference and solution() kernels DIFFER  ***")
        print(f"  max |y_ref - y_sol| = {max_diff:.2e}")
        print(sep)

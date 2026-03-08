
# --- Legacy identity feature map (kept for reference) ---
class TrainiumLinearAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads

        # Project and split heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, h, self.head_dim).transpose(1, 2), qkv)

        # 1. Identity Feature Map (No Softmax)
        q = q * self.scale

        # 2. Linear Attention via Associative Property
        kv_state = torch.matmul(k.transpose(-1, -2), v)
        out = torch.matmul(q, kv_state)

        # Recombine heads
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.to_out(out)
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm

device = xm.xla_device()

device = 'xla'
# class TrainiumLinearAttention(nn.Module):
#     def __init__(self, dim, heads=8):
#         super().__init__()
#         self.heads = heads
#         self.head_dim = dim // heads
#         self.scale = self.head_dim ** -0.5
        
#         self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.to_out = nn.Linear(dim, dim)

#     def forward(self, x):
#         b, n, d = x.shape
#         h = self.heads
        
#         # Project and split heads
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: t.view(b, n, h, self.head_dim).transpose(1, 2), qkv)

#         # 1. Identity Feature Map (No Softmax)
#         # Apply scaling to Q
#         q = q * self.scale

#         # 2. Linear Attention via Associative Property
#         # Complexity: O(N * d^2) instead of O(N^2 * d)
#         # We compute (K^T @ V) first
#         # k: [b, h, n, d_h] -> k.transpose: [b, h, d_h, n]
#         # v: [b, h, n, d_h]
#         kv_state = torch.matmul(k.transpose(-1, -2), v) # [b, h, d_h, d_h]
        
#         # 3. Multiply Q by the state
#         out = torch.matmul(q, kv_state) # [b, h, n, d_h]

#         # Recombine heads
#         out = out.transpose(1, 2).reshape(b, n, d)
#         return self.to_out(out)

# # --- Execution Setup for Trainium 1 ---
# device = "xla" # Standard for Neuron/Trainium
# model = TrainiumLinearAttention(dim=512).to(device)

# # Use torch.compile to let the Neuron compiler fuse these matmuls
# # This is essential for preventing the 'Memory Wall' on Trainium
# model = torch.compile(model, backend="openxla")

# dummy_input = torch.randn(2, 2048, 512).to(device)
# output = model(dummy_input)
# print(f"Output shape: {output.shape}")

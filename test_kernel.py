import torch
torch.ops.load_library("gemm_ext.so")
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

M, N, K = 1024, 1024, 1024
# A = torch.arange(M * K, dtype=torch.bfloat16, device='cuda').reshape(M, K)
# B = torch.ones(K, N, dtype=torch.bfloat16, device='cuda')
# A = torch.ones(M, K, dtype=torch.bfloat16, device='cuda')
# B = torch.arange(N * K, dtype=torch.bfloat16, device='cuda').reshape(K, N)
# B = B.t().contiguous().t()
# C = torch.ops.gemm_ext.gemm_caller(A, B, 1, 1.0, 0.0).cpu().int().numpy()
# print("--------------------------------")
# print(C)

A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
B = B.t().contiguous().t()
C = torch.ops.gemm_ext.gemm_caller(A, B, 1, 1.0, 0.0)
C_gt = A @ B
assert torch.allclose(C, C_gt)
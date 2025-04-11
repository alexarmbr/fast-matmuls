import torch
torch.ops.load_library("gemm_ext.so")
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# M, N, K = 128, 256, 64
# A = torch.arange(M * K, dtype=torch.bfloat16, device='cuda').reshape(M, K)
# B = torch.ones(K, N, dtype=torch.bfloat16, device='cuda')
# # A = torch.ones(M, K, dtype=torch.bfloat16, device='cuda')
# # B = torch.arange(N * K, dtype=torch.bfloat16, device='cuda').reshape(K, N)
# B = B.t().contiguous().t()
# # C = torch.ops.gemm_ext.gemm_caller(A, B, 2, 1.0, 0.0).cpu().int().numpy()
# C = torch.ops.gemm_ext.gemm_caller(A, B, 2, 1.0, 0.0)
# C_gt = A @ B
# assert torch.allclose(C, C_gt)



# import pdb; pdb.set_trace()
# print("--------------------------------")
# print("first column of C:")
# print(C[:, 0])
# print("--------------------------------")
# print("first row of C:")
# print(C[0, :])
# print("--------------------------------")

M, N, K = 2048, 2048, 2048
A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
B = B.t().contiguous().t()

C = torch.ops.gemm_ext.gemm_caller(A, B, 3)
C_gt = A @ B
assert torch.allclose(C, C_gt)

C = torch.ops.gemm_ext.gemm_caller(A, B, 2)
C_gt = A @ B
assert torch.allclose(C, C_gt)
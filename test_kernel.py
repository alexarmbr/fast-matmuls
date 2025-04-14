import torch
torch.ops.load_library("gemm_ext.so")
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


def test_kernels(M, N, K, kernel_id, seed=0):
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    B = B.t().contiguous().t()
    C = torch.ops.gemm_ext.gemm_caller(A, B, kernel_id)
    C_gt = A @ B
    

for i in range(10):
    seed = 473395
    test_kernels(2048, 2048, 2048, 1, seed)
    test_kernels(2048, 2048, 2048, 2, seed)
    test_kernels(2048, 2048, 2048, 3, seed)
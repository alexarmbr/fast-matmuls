#!/bin/bash

#inline python string
program="
import torch
import os

torch.ops.load_library('gemm_ext.so')

M, N, K = 8192, 8192, 8192
A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
B = B.t().contiguous().t()
C = torch.ops.gemm_ext.gemm_caller(A, B, 1, 1.0, 0.0)
"

ncu -f -o profiling/kernel --set full python -c "$program"
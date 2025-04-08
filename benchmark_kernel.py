import torch
torch.ops.load_library("gemm_ext.so")
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def benchmark_kernel(M, N, K, n_iters=20):
    
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    B = B.t().contiguous().t()
    
    # warmup
    for i in range(5):
        _ = torch.ops.gemm_ext.gemm_caller(A, B, 1, 1.0, 0.0)
    
    # benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(n_iters):
        _ = torch.ops.gemm_ext.gemm_caller(A, B, 1, 1.0, 0.0)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    return elapsed_ms / n_iters

def compute_tflops_per_second(M, N, K, elapsed_ms):
    flops = 2 * M * N * K
    flops_per_second = flops / (elapsed_ms * 1e-3)
    tflops_per_second = flops_per_second / 1e12
    return tflops_per_second

if __name__ == "__main__":
    benchmark_dimensions = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]

    benchmark_data = []
    for M, N, K in benchmark_dimensions:
        elapsed_ms = benchmark_kernel(M, N, K)
        tflops_per_second = compute_tflops_per_second(M, N, K, elapsed_ms)
        print("--------------------------------")
        print(f"M: {M}, N: {N}, K: {K}")
        print(f"Time taken: {elapsed_ms} ms, TFLOPs/sec: {tflops_per_second}")
        print("--------------------------------")
        benchmark_data.append((M, tflops_per_second))
    
    # make a plot of M vs TFLOPs/sec
    import matplotlib.pyplot as plt
    import numpy as np

    M_values = [data[0] for data in benchmark_data]
    tflops_values = [data[1] for data in benchmark_data]

    plt.figure(figsize=(10, 6))
    plt.plot(M_values, tflops_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Matrix Size (M)')
    plt.ylabel('TFLOPs/sec')

    # save to disk
    plt.savefig('benchmark_kernel.png')
    

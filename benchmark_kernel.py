import torch
torch.ops.load_library("gemm_ext.so")
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def benchmark_kernel(M, N, K, kernel_id, n_iters=20):
    
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    B = B.t().contiguous().t()
    
    # warmup
    for i in range(5):
        _ = torch.ops.gemm_ext.gemm_caller(A, B, kernel_id)
    torch.cuda.synchronize()

    # benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(n_iters):
        _ = torch.ops.gemm_ext.gemm_caller(A, B, kernel_id)
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
        # (512, 512, 512),
        # (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]

    benchmark_data = []
    for kernel_id in range(1, 4):
        for M, N, K in benchmark_dimensions:
            elapsed_ms = benchmark_kernel(M, N, K, kernel_id)
            tflops_per_second = compute_tflops_per_second(M, N, K, elapsed_ms)
            print("--------------------------------")
            print(f"M: {M}, N: {N}, K: {K}")
            print(f"Time taken: {elapsed_ms} ms, TFLOPs/sec: {tflops_per_second}")
            print("--------------------------------")
            benchmark_data.append((M, tflops_per_second, kernel_id))
    

    import matplotlib.pyplot as plt
    import numpy as np

    # Extract data for each kernel
    kernel_1_data = [data for data in benchmark_data if data[2] == 1]
    kernel_2_data = [data for data in benchmark_data if data[2] == 2]
    kernel_3_data = [data for data in benchmark_data if data[2] == 3]

    print("kernel_1_data")
    print(kernel_1_data)

    print("kernel_2_data")
    print(kernel_2_data)

    print("kernel_3_data")
    print(kernel_3_data)

    # plot all kernels on the same plot
    plt.figure(figsize=(12, 8))
    plt.plot([data[0] for data in kernel_1_data], [data[1] for data in kernel_1_data], marker='o', linestyle='-', color='b', label='Kernel 1')
    plt.plot([data[0] for data in kernel_2_data], [data[1] for data in kernel_2_data], marker='o', linestyle='-', color='r', label='Kernel 2')
    plt.plot([data[0] for data in kernel_3_data], [data[1] for data in kernel_3_data], marker='o', linestyle='-', color='g', label='Kernel 3')
    plt.xlabel('Matrix Size (M)')
    plt.ylabel('TFLOPs/sec')
    plt.legend()
    plt.savefig('benchmark_kernel.png')
    
    
        


    

#include "ATen/ATen.h" // @manual
#include "torch/extension.h" // @manual

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_bf16.h"

using bf16 = __nv_bfloat16;

// Forward declaration of CUDA functions
namespace kernel1 {
void launch(
    void* A_device,
    void* B_device,
    void* C_device,
    unsigned int M,
    unsigned int N,
    unsigned int K);
} // namespace kernel1

namespace kernel2 {
void launch(
    void* A_device,
    void* B_device,
    void* C_device,
    unsigned int M,
    unsigned int N,
    unsigned int K);
} // namespace kernel2

namespace kernel3 {
void launch(
    void* A_device,
    void* B_device,
    void* C_device,
    unsigned int M,
    unsigned int N,
    unsigned int K);
} // namespace kernel3

// C++ wrapper function that will be called from Python
torch::Tensor gemm_caller(
    torch::Tensor A,
    torch::Tensor B,
    int64_t kernel_id) {
    
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);
    
    // C has the same dtype and device as A
    torch::Tensor C = A.new_empty({M, N});
    
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(B.scalar_type() == torch::kBFloat16, "B must be bfloat16");
    TORCH_CHECK(C.scalar_type() == torch::kBFloat16, "C must be bfloat16");
    
    // Check dimensions
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(C.dim() == 2, "C must be 2D");
    

    // im-trans-a and im-trans-b arguments to the mma instruction are False
    // this means that both A and B are k-major (contiguous on the k dimensions)
    // so A is (m, k) as usual
    // B is (n, k) and row major
    // or alternatively B is (k, n) but column major
    // both are equivalent from the perspective of linear memory
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#shared-memory-matrix-layout
    TORCH_CHECK(A.size(1) == B.size(0), "Dimension mismatch: B.size(1) must equal A.size(1)");
    TORCH_CHECK(A.stride(1) == 1, "A must be contiguous in the second dimension");
    TORCH_CHECK(B.stride(0) == 1, "B must be contiguous in the first dimension");
    
    if (kernel_id == 1) {
        kernel1::launch(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K);
    }
    else if (kernel_id == 2) {
        kernel2::launch(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K);
    }
    else if (kernel_id == 3) {
        kernel3::launch(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K);
    }
    else {
        TORCH_CHECK(false, "Invalid kernel ID");
    }
    return C;
}

TORCH_LIBRARY(gemm_ext, m) {
    m.def("gemm_caller", &gemm_caller);
}
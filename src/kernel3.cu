#include <cuda.h>
#include <cuda/barrier>
#include <mma.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <iostream>

namespace kernel3 {

using bf16 = __nv_bfloat16;
using barrier = cuda::barrier<cuda::thread_scope_block>;

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint32_t x)
{
  return (x & 0x3FFFF) >> 4;
}

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void *pointer) {
    uint32_t address;
    asm("{\n\t"
        "  .reg .u64 u64addr;\n\t"
        "  cvta.to.shared.u64 u64addr, %1;\n\t"
        "  cvt.u32.u64 %0, u64addr;\n\t"
        "}"
        : "=r"(address)
        : "l"(pointer));
    return address;
  }

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
template <unsigned int smem_width_bytes>
__device__ uint64_t make_smem_desc(bf16* ptr) {
  constexpr uint64_t leading_dim_byte_offset = 16;
  constexpr uint64_t stride_dim_byte_offset = smem_width_bytes;
  constexpr uint64_t swizzle_128b = 1ull;
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  return matrix_descriptor_encode(addr) |
      (matrix_descriptor_encode(leading_dim_byte_offset) << 16) |
      (matrix_descriptor_encode(stride_dim_byte_offset * 8) << 32) |
      (swizzle_128b << 62);
}

template <unsigned int BK_dim>
__device__ __forceinline__ void wgmma_m64n256k16_f32_bf16_bf16(float D[128], bf16* A_block_smem, bf16* B_block_smem){
    uint64_t A_desc = make_smem_desc<BK_dim * sizeof(bf16)>(A_block_smem);
    uint64_t B_desc = make_smem_desc<BK_dim * sizeof(bf16)>(B_block_smem);

    // uint64_t A_desc = make_smem_desc(A_block_smem);
    // uint64_t B_desc = make_smem_desc(B_block_smem);
    
    asm volatile (
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31,"
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,"
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63,"
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79,"
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95,"
        "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111,"
        "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, " // D (accumulator registers)
        
        "%128, %129, " // A_desc, B_desc (shared memory descriptors)

        "1, 1, 1, 0, 0;" //  scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b
        // scale-d=1: compute D = A * B + D rather than D = A * B
        // imm-scale-a=1: compute A = A * 1.0f (no scaling, A can optionally be negated if you pass -1)
        // imm-scale-b=1: compute B = B * 1.0f (no scaling, B can optionally be negated if you pass -1)
        // imm-trans-a=0: A is k-major in shared memory
        // imm-trans-b=0: B is k-major in shared memory
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7]), "+f"(D[8]), "+f"(D[9]), "+f"(D[10]), "+f"(D[11]), "+f"(D[12]), "+f"(D[13]), "+f"(D[14]), "+f"(D[15]),
            "+f"(D[16]), "+f"(D[17]), "+f"(D[18]), "+f"(D[19]), "+f"(D[20]), "+f"(D[21]), "+f"(D[22]), "+f"(D[23]), "+f"(D[24]), "+f"(D[25]), "+f"(D[26]), "+f"(D[27]), "+f"(D[28]), "+f"(D[29]), "+f"(D[30]), "+f"(D[31]),
            "+f"(D[32]), "+f"(D[33]), "+f"(D[34]), "+f"(D[35]), "+f"(D[36]), "+f"(D[37]), "+f"(D[38]), "+f"(D[39]), "+f"(D[40]), "+f"(D[41]), "+f"(D[42]), "+f"(D[43]), "+f"(D[44]), "+f"(D[45]), "+f"(D[46]), "+f"(D[47]),
            "+f"(D[48]), "+f"(D[49]), "+f"(D[50]), "+f"(D[51]), "+f"(D[52]), "+f"(D[53]), "+f"(D[54]), "+f"(D[55]), "+f"(D[56]), "+f"(D[57]), "+f"(D[58]), "+f"(D[59]), "+f"(D[60]), "+f"(D[61]), "+f"(D[62]), "+f"(D[63]),
            "+f"(D[64]), "+f"(D[65]), "+f"(D[66]), "+f"(D[67]), "+f"(D[68]), "+f"(D[69]), "+f"(D[70]), "+f"(D[71]), "+f"(D[72]), "+f"(D[73]), "+f"(D[74]), "+f"(D[75]), "+f"(D[76]), "+f"(D[77]), "+f"(D[78]), "+f"(D[79]),
            "+f"(D[80]), "+f"(D[81]), "+f"(D[82]), "+f"(D[83]), "+f"(D[84]), "+f"(D[85]), "+f"(D[86]), "+f"(D[87]), "+f"(D[88]), "+f"(D[89]), "+f"(D[90]), "+f"(D[91]), "+f"(D[92]), "+f"(D[93]), "+f"(D[94]), "+f"(D[95]),
            "+f"(D[96]), "+f"(D[97]), "+f"(D[98]), "+f"(D[99]), "+f"(D[100]), "+f"(D[101]), "+f"(D[102]), "+f"(D[103]), "+f"(D[104]), "+f"(D[105]), "+f"(D[106]), "+f"(D[107]), "+f"(D[108]), "+f"(D[109]), "+f"(D[110]), "+f"(D[111]),
            "+f"(D[112]), "+f"(D[113]), "+f"(D[114]), "+f"(D[115]), "+f"(D[116]), "+f"(D[117]), "+f"(D[118]), "+f"(D[119]), "+f"(D[120]), "+f"(D[121]), "+f"(D[122]), "+f"(D[123]), "+f"(D[124]), "+f"(D[125]), "+f"(D[126]), "+f"(D[127])
            : "l"(A_desc), "l"(B_desc)
    );
}

template <unsigned int SMEM_HEIGHT, unsigned int SMEM_WIDTH>
void __createTensorMapHost(bf16* tensor_ptr, unsigned int gmem_height, unsigned int gmem_width, CUtensorMap* tensor_map)
{
  constexpr uint32_t rank = 2;
  uint64_t gmem_size[2] = {uint64_t(gmem_width), uint64_t(gmem_height)};
  uint64_t gmem_stride[1] = {sizeof(bf16) * gmem_width};
  uint32_t smem_size[2] = {uint32_t(SMEM_WIDTH), uint32_t(SMEM_HEIGHT)};
  uint32_t smem_stride[2] = {1, 1};

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
    tensor_map,                // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    rank,                       // cuuint32_t tensorRank,
    tensor_ptr,                 // void *globalAddress,
    gmem_size,                       // const cuuint64_t *globalDim,
    gmem_stride,                     // const cuuint64_t *globalStrides,
    smem_size,                   // const cuuint32_t *boxDim,
    smem_stride,                // const cuuint32_t *elementStrides,
    // Interleave patterns can be used to accelerate loading of values that
    // are less than 4 bytes long.
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    // Swizzling can be used to avoid shared memory bank conflicts.
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    // L2 Promotion can be used to widen the effect of a cache-policy to a wider
    // set of L2 cache lines.
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    // Any element that is outside of bounds will be set to zero by the TMA transfer.
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );

  if (res != CUDA_SUCCESS){
    printf("Error encoding tensor map: %d\n", res);
  }
  assert(res == CUDA_SUCCESS);
}

template <unsigned int SMEM_HEIGHT, unsigned int SMEM_WIDTH>
void createTensorMap(bf16* tensor_ptr, unsigned int gmem_height, unsigned int gmem_width, CUtensorMap* tensor_map_device)
{
  CUtensorMap tensor_map_host;
  __createTensorMapHost<SMEM_HEIGHT, SMEM_WIDTH>(tensor_ptr, gmem_height, gmem_width, &tensor_map_host);
  CUDA_CHECK(cudaMemcpy(tensor_map_device, &tensor_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
}


struct block_coordinates {
  unsigned int block_m;
  unsigned int block_n;
};

template <unsigned int BM_dim, unsigned int BN_dim, unsigned int TILE_M = 16, unsigned int TILE_N = 8>
__device__ block_coordinates get_block_coordinates(unsigned int iteration, unsigned int M, unsigned int N){
  unsigned int block_m = blockIdx.x % TILE_N;
  unsigned int block_n = blockIdx.x / TILE_N;
  
  constexpr unsigned int n_elements_per_group = TILE_N * BN_dim;
  unsigned int iterations_n = N / n_elements_per_group;
  
  unsigned int n = block_m + (iteration % iterations_n) * TILE_N;
  unsigned int m = block_n + (iteration / iterations_n) * TILE_M;
  assert(n < (N / BN_dim));
  assert(m < (M / BM_dim));
  
  return {m, n};
}


template <unsigned int BM_dim,
unsigned int BN_dim,
unsigned int BK_dim,
unsigned int QSIZE>
__global__ void
matmul_kernel(
  const CUtensorMap* tensorMapA,
  const CUtensorMap* tensorMapB,
  bf16* C,
  const unsigned int M,
  const unsigned int N,
  const unsigned int K)
{

  constexpr unsigned int WGMMA_M = 64;
  constexpr unsigned int WGMMA_N = 256;
  constexpr unsigned int WGMMA_K = 16;

  // launch a total of 128 blocks and divide them into 16x8 grid
  bool is_producer = threadIdx.x < 128;

  // __shared__ alignas(128) bf16 A_block_smem[BM_dim*BK_dim*QSIZE];
  // __shared__ alignas(128) bf16 B_block_smem[BN_dim*BK_dim*QSIZE];
  extern __shared__ __align__(128) bf16 smem[];
  bf16* A_block_smem = smem;
  bf16* B_block_smem = smem + BM_dim * BK_dim * QSIZE;
  constexpr unsigned int pipelineStageElements = BM_dim * BK_dim + BN_dim * BK_dim;
  constexpr unsigned int pipelineStageNumBytes = pipelineStageElements * sizeof(bf16); // total amount of bytes read from gmem to smem in one pipeline stage
  
  // this pragma suppresses the warning about static variables with dynamic initialization
  #pragma nv_diag_suppress static_var_with_dynamic_init

  __shared__ barrier empty_barrier[QSIZE], full_barrier[QSIZE];

  if (threadIdx.x == 0) {
    for (int i = 0; i < QSIZE; i++){
      init(&empty_barrier[i], 1 + 2 * 128); // 1 producer thread, plus all consumer threads
      init(&full_barrier[i], 1 + 2 * 128);
    }

    // this synchronizes with the tensor memory accelerator (TMA)
    // the TMA is a hardware unit that operates asychronously with respect to other stuff happening on the SM
    cuda::device::experimental::fence_proxy_async_shared_cta();
  }

  // sychronize so that the initialized barrier is visible to all threads
  __syncthreads();
  
  const unsigned int total_iterations = ((M / BM_dim) * (N / BN_dim)) / gridDim.x;
    if (is_producer){
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(24));
      int buffer_stage = 0;

      for (unsigned int persistent_kernel_iter = 0; persistent_kernel_iter < total_iterations; persistent_kernel_iter++){
        block_coordinates block_coords = get_block_coordinates<BM_dim, BN_dim>(persistent_kernel_iter, M, N);

          if (threadIdx.x == 0){
            for (int block_k = 0; block_k < K / BK_dim; block_k++){
              // producer arrives and waits at the empty barrier for current pipeline stage
              // once the consumer arrives at the same empty barrier (signifying that it has consumed this pipeline stage)
              // producer can initiate the TMA copy that will write to this pipeline stage
              empty_barrier[buffer_stage].wait(empty_barrier[buffer_stage].arrive());

              // copy A and B from gmem to smem
              cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(&A_block_smem[buffer_stage * (BM_dim * BK_dim)], tensorMapA, block_k * BK_dim, block_coords.block_m * BM_dim, full_barrier[buffer_stage]);
              cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(&B_block_smem[buffer_stage * (BN_dim * BK_dim)], tensorMapB, block_k * BK_dim, block_coords.block_n * BN_dim, full_barrier[buffer_stage]);
              
              // producer arrives at the full barrier for current pipeline stage and sets the transaction count to BM * BK + BN * BK bytes
              // anyone waiting at this full_barrier will only proceed once this many bytes have been transferred from gmem to smem
              barrier::arrival_token _ = cuda::device::barrier_arrive_tx(full_barrier[buffer_stage], 1, pipelineStageNumBytes);
              buffer_stage = (buffer_stage + 1) % QSIZE;
            }
          }
        }
    }

    // consumer
    else {
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(240));
      int buffer_stage = 0;
      float D_reg[128];
      memset(D_reg, 0, sizeof(D_reg));
      const unsigned int consumer_idx = (threadIdx.x / 128) - 1;

      // initialize the empty barrier for all pipeline stages with consumer arriving
      for (int i = 0; i < QSIZE; i++){
        barrier::arrival_token _ = empty_barrier[i].arrive();
      }
      
      for (unsigned int persistent_kernel_iter = 0; persistent_kernel_iter < total_iterations; persistent_kernel_iter++){
      block_coordinates block_coords = get_block_coordinates<BM_dim, BN_dim>(persistent_kernel_iter, M, N);

      for (int block_k = 0; block_k < K / BK_dim; block_k++)
      {
        full_barrier[buffer_stage].wait(full_barrier[buffer_stage].arrive());

        warpgroup_arrive();
        
        bf16* A_stage_smem = A_block_smem + buffer_stage * (BM_dim * BK_dim);
        bf16* B_stage_smem = B_block_smem + buffer_stage * (BN_dim * BK_dim);
        bf16* A_warpgroup_smem = A_stage_smem + consumer_idx * WGMMA_M * BK_dim;
        
        #pragma unroll
        for (int i = 0; i < BK_dim / WGMMA_K; i++){
          int offset = i * WGMMA_K;
          wgmma_m64n256k16_f32_bf16_bf16<BK_dim>(D_reg, A_warpgroup_smem + offset, B_stage_smem + offset);
        }

        wgmma_commit_group();
        wgmma_wait_group<0>();

        barrier::arrival_token _ = empty_barrier[buffer_stage].arrive();
        buffer_stage = (buffer_stage + 1) % QSIZE;
      }
    
      bf16* C_block = C + (block_coords.block_m * BM_dim + consumer_idx * WGMMA_M) * N + block_coords.block_n * BN_dim;

      #define OUT_IDX(i, j) (i) * N + (j)
      int thread = threadIdx.x % 128;
      int warp = thread / 32;
      int lane = thread % 32;
      int row = (warp * 16) + (lane / 4);
      int col = (thread % 4) * 2;
      for (int column_group = 0; column_group < WGMMA_N / 16; column_group++)
      {
        C_block[OUT_IDX(row, col)] = __float2bfloat16(D_reg[column_group * 8]);
        C_block[OUT_IDX(row, col + 1)] = __float2bfloat16(D_reg[column_group * 8 + 1]);
        C_block[OUT_IDX(row + 8, col)] = __float2bfloat16(D_reg[column_group * 8 + 2]);
        C_block[OUT_IDX(row + 8, col + 1)] = __float2bfloat16(D_reg[column_group * 8 + 3]);
        C_block[OUT_IDX(row, col + 8)] = __float2bfloat16(D_reg[column_group * 8 + 4]);
        C_block[OUT_IDX(row, col + 9)] = __float2bfloat16(D_reg[column_group * 8 + 5]);
        C_block[OUT_IDX(row + 8, col + 8)] = __float2bfloat16(D_reg[column_group * 8 + 6]);
        C_block[OUT_IDX(row + 8, col + 9)] = __float2bfloat16(D_reg[column_group * 8 + 7]);
        col += 16;
      }
      #undef OUT_IDX
    }
  }
}

CUtensorMap* A_tensor_map_device = nullptr;
CUtensorMap* B_tensor_map_device = nullptr;
bf16* A_device_bf16_ = nullptr;
bf16* B_device_bf16_ = nullptr;
bf16* C_device_bf16_ = nullptr;

void launch(void* A_device, void* B_device, void* C_device, unsigned int M, unsigned int N, unsigned int K)
{
    // block scheduling logic only supports M, N, K >= 2048
    assert(M >= 2048);
    assert(N >= 2048);
    assert(K >= 2048);

    constexpr unsigned int num_consumer_warpgroups = 2;
    constexpr unsigned int BM_dim = 64 * num_consumer_warpgroups;
    constexpr unsigned int BN_dim = 256;
    constexpr unsigned int BK_dim = 64;
    constexpr unsigned int QSIZE = 4;
    constexpr unsigned int shmemNumBytes = (BM_dim * BK_dim + BK_dim * BN_dim) * sizeof(bf16) * QSIZE;

    static_assert(shmemNumBytes <= 227000, "sm90 has a max of 227000 bytes of dynamic shared memory");

    bf16* A_device_bf16 = reinterpret_cast<bf16*>(A_device);
    bf16* B_device_bf16 = reinterpret_cast<bf16*>(B_device);
    bf16* C_device_bf16 = reinterpret_cast<bf16*>(C_device);

    if (A_tensor_map_device == nullptr || B_tensor_map_device == nullptr || A_device_bf16_ != A_device_bf16 || B_device_bf16_ != B_device_bf16 || C_device_bf16_ != C_device_bf16){
      
      if (A_tensor_map_device == nullptr){
        CUDA_CHECK(cudaMalloc(&A_tensor_map_device, sizeof(CUtensorMap)));
      }
      
      if (B_tensor_map_device == nullptr){
        CUDA_CHECK(cudaMalloc(&B_tensor_map_device, sizeof(CUtensorMap)));
      }
      
      A_device_bf16_ = A_device_bf16;
      B_device_bf16_ = B_device_bf16;
      C_device_bf16_ = C_device_bf16;
      createTensorMap<BM_dim, BK_dim>(A_device_bf16, M, K, A_tensor_map_device);
      createTensorMap<BN_dim, BK_dim>(B_device_bf16, N, K, B_tensor_map_device);
    }

    CUDA_CHECK(cudaFuncSetAttribute(matmul_kernel<BM_dim, BN_dim, BK_dim, QSIZE>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    shmemNumBytes));

    dim3 gridDimension(128);
    dim3 blockDimension(128 * (num_consumer_warpgroups + 1));

    matmul_kernel
    <BM_dim, BN_dim, BK_dim, QSIZE>
    <<<gridDimension, blockDimension, shmemNumBytes>>>(
        A_tensor_map_device,
        B_tensor_map_device,
        C_device_bf16,
        M,
        N,
        K
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
}


} // namespace kernel3

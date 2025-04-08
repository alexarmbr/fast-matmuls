

# 

def shmem_bytes(M, N, K):
    return (M*K + K*N) * 2

def count_registers(M, N, K, mma_shape, num_warpgroups=1):
    
    threads_per_warpgroup = 128
    num_threads = threads_per_warpgroup * num_warpgroups
    
    if mma_shape == "m64n256k16":
        mma_m = 64
        mma_n = 256
        registers_per_mma_tile = 128
        mma_k = 16
    elif mma_shape == "m64n128k16":
        mma_m = 64
        mma_n = 128
        registers_per_mma_tile = 64
        mma_k = 16
    elif mma_shape == "m64n64k16":
        mma_m = 64
        mma_n = 64
        registers_per_mma_tile = 32
        mma_k = 16
    else:
        assert False
    
    assert M % mma_m == 0
    assert N % mma_n == 0
    assert K % mma_k == 0
    
    M_warpgroup = M // num_warpgroups
    
    m_tiles = M_warpgroup // mma_m
    n_tiles = N // mma_n
    k_tiles = K // mma_k

    registers_per_thread = m_tiles * n_tiles * registers_per_mma_tile
    registers_per_block = registers_per_thread * num_threads
    return registers_per_thread, registers_per_block



configs = [
    (64, 256, 64, "m64n256k16", 1),
    (128, 256, 64, "m64n256k16", 1),
    (128, 256, 64, "m64n256k16", 2),
]

for (BM, BN, BK, mma_shape, num_warpgroups) in configs:
    print(f"M: {BM}, N: {BN}, K: {BK}, mma_shape: {mma_shape}, num_warpgroups: {num_warpgroups}")
    print(f"SHMEM: {shmem_bytes(BM, BN, BK)} bytes")
    registers_per_thread, registers_per_block = count_registers(BM, BN, BK, "m64n256k16", num_warpgroups)
    print(f"REGISTERS: {registers_per_thread} per thread, {registers_per_block} per block")
    print("----------------------------------------")


    

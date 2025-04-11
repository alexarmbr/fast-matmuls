# logic for threadblock layout and iteration in persistent kernel
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

M, N, K = 4096, 4096, 4096
BM, BN, BK = 128, 256, 64

# use 128 of the 132 SMs so that we can have a 16 x 8 tile
num_sm = 128

# tile dimensions
group_n = 8
group_m = 16

assert group_n * group_m == 128
assert M >= 2048
assert N >= 2048
assert K >= 2048

def get_block_coordinates(i, block_idx):
    block_m = block_idx % group_n
    block_n = block_idx // group_n
    
    # how many row elements in a group
    n_elements_per_group = group_n * BN # static
    m_elements_per_group = group_m * BM

    # how many groups fit in a row?
    iterations_n = N // n_elements_per_group
    iterations_m = M // m_elements_per_group

    # every time group advances, block_n advances by 8
    n = block_idx % 8 + (i % iterations_n) * group_n
    m = block_idx // 8 + (i // iterations_n) * group_m
    return m, n


blocks_m = M // BM
blocks_n = N // BN
assert ((blocks_m * blocks_n)) % num_sm == 0
num_iterations = (((blocks_m * blocks_n)) // num_sm)

print(f"num_iterations: {num_iterations}")

for i in range(num_iterations):
    print(f"---------  iteration {i} ---------")
    
    sm_grid = np.zeros((blocks_m, blocks_n), dtype=np.int32)
    
    for block_idx in range(num_sm):
        m, n = get_block_coordinates(i, block_idx)
        sm_grid[m, n] = block_idx
    assert sm_grid.sum() == sum(range(num_sm))

    print(sm_grid)











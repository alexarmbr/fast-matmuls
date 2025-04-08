#   int thread = threadIdx.x % 128;
#   int row = ((thread / 32) * 16) + ((thread % 32) / 4);
#   int col = (((thread % 32) / 2) * 2) % 8;
import numpy as np


# turn off print limit
# set column width to infinite
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

a = np.zeros((64, 16))

for thread in range(128):
    row = ((thread // 32) * 16) + ((thread % 32) // 4)
    col = (thread % 4) * 2

    print(f"thread: {thread}, row: {row}, col: {col}")
    
    a[row, col] = 1 * thread
    a[row, col+1] = 2 * thread
    
    a[row+8, col] = 3 * thread
    a[row+8, col+1] = 4 * thread

    a[row, col+8] = 4 * thread
    a[row, col+9] = 5 * thread

    a[row+8, col+8] = 6 * thread
    a[row+8, col+9] = 7 * thread

print(a)

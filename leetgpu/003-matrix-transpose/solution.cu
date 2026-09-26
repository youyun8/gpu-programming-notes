// Matrix Transpose (LeetGPU)
// https://leetgpu.com/challenges/matrix-transpose
#include <cuda_runtime.h>

constexpr int kTile = 32;
constexpr int kRowsPerPass = 8;  // block is 32 x 8, each thread moves 4 elements

__global__ void transposeTiled(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[kTile][kTile + 1];  // +1 column avoids bank conflicts

    // Load a 32 x 32 input tile: each warp reads one input row segment (coalesced),
    // 8 rows per pass, 4 passes.
    int x = blockIdx.x * kTile + threadIdx.x;  // column in input
    int y = blockIdx.y * kTile + threadIdx.y;  // row in input
    for (int j = 0; j < kTile; j += kRowsPerPass) {
        if (x < cols && y + j < rows) tile[threadIdx.y + j][threadIdx.x] = input[static_cast<size_t>(y + j) * cols + x];
    }
    // The whole tile is in shared memory.
    __syncthreads();

    // Swap block coordinates so writes to output are coalesced as well.
    // Write the transposed tile: lanes now read a tile column (padding keeps it
    // conflict-free) and write one output row segment (coalesced).
    x = blockIdx.y * kTile + threadIdx.x;  // column in output (= row in input)
    y = blockIdx.x * kTile + threadIdx.y;  // row in output (= column in input)
    for (int j = 0; j < kTile; j += kRowsPerPass) {
        if (x < rows && y + j < cols) output[static_cast<size_t>(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    // 32 x 8 threads per 32 x 32 tile.
    const dim3 block(kTile, kRowsPerPass);
    const dim3 grid((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
    transposeTiled<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

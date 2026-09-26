// 2D Subarray Sum (LeetGPU)
// https://leetgpu.com/challenges/2d-subarray-sum
//
// Sum of the rectangle [S_ROW..E_ROW] x [S_COL..E_COL] of an N x M int matrix.
// The rectangle is flattened to one index space; consecutive threads read
// consecutive columns of a row (coalesced).
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 2048;

__global__ void rectSum(const int* input, int* output, int m, int s_row, int s_col, int rows, int cols) {
    const long long total = static_cast<long long>(rows) * cols;
    int local = 0;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < total;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int r = static_cast<int>(i / cols);
        const int c = static_cast<int>(i % cols);
        local += input[static_cast<size_t>(s_row + r) * m + s_col + c];
    }
    local = __reduce_add_sync(0xffffffffu, local);
    if (threadIdx.x % 32 == 0 && local) atomicAdd(output, local);
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    cudaMemset(output, 0, sizeof(int));
    const int rows = E_ROW - S_ROW + 1;
    const int cols = E_COL - S_COL + 1;
    long long blocks = (static_cast<long long>(rows) * cols + kBlockSize - 1) / kBlockSize;
    blocks = blocks > kMaxBlocks ? kMaxBlocks : blocks;
    rectSum<<<static_cast<int>(blocks), kBlockSize>>>(input, output, M, S_ROW, S_COL, rows, cols);
    cudaDeviceSynchronize();
}

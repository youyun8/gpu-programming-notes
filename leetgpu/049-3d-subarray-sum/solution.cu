// 3D Subarray Sum (LeetGPU)
// https://leetgpu.com/challenges/3d-subarray-sum
//
// Sum of the box [S_DEP..E_DEP] x [S_ROW..E_ROW] x [S_COL..E_COL] of an
// N x M x K int volume, flattened to one index space (column fastest).
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 2048;

__global__ void boxSum(const int* input, int* output, int m, int k, int s_dep, int s_row, int s_col, int deps,
                       int rows, int cols) {
    const long long total = static_cast<long long>(deps) * rows * cols;
    int local = 0;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < total;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int c = static_cast<int>(i % cols);
        const int r = static_cast<int>((i / cols) % rows);
        const int d = static_cast<int>(i / (static_cast<long long>(cols) * rows));
        local += input[(static_cast<size_t>(s_dep + d) * m + s_row + r) * k + s_col + c];
    }
    local = __reduce_add_sync(0xffffffffu, local);
    if (threadIdx.x % 32 == 0 && local) atomicAdd(output, local);
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP, int S_ROW, int E_ROW,
                      int S_COL, int E_COL) {
    cudaMemset(output, 0, sizeof(int));
    const int deps = E_DEP - S_DEP + 1;
    const int rows = E_ROW - S_ROW + 1;
    const int cols = E_COL - S_COL + 1;
    long long blocks = (static_cast<long long>(deps) * rows * cols + kBlockSize - 1) / kBlockSize;
    blocks = blocks > kMaxBlocks ? kMaxBlocks : blocks;
    boxSum<<<static_cast<int>(blocks), kBlockSize>>>(input, output, M, K, S_DEP, S_ROW, S_COL, deps, rows, cols);
    cudaDeviceSynchronize();
}

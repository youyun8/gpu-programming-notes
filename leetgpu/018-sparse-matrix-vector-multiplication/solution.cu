// Sparse Matrix-Vector Multiplication (LeetGPU)
// https://leetgpu.com/challenges/sparse-matrix-vector-multiplication
//
// A is given densely (60-70% zeros) in row-major order, so there is no index
// structure to exploit: the kernel is a bandwidth-bound GEMV. One warp per
// row: lanes stride over the columns (coalesced 128-byte reads of A), x is
// small enough (<= 40 KB) to stay in L1/L2, and a shuffle reduction produces y.
// Skipping zero entries would save FLOPs, not bytes, so it does not help.
#include <cuda_runtime.h>

constexpr int kWarpsPerBlock = 8;

__global__ void gemvWarpPerRow(const float* a, const float* x, float* y, int m, int n) {
    // One warp per row of the (dense-stored) matrix.
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * kWarpsPerBlock + threadIdx.x / 32;
    if (row >= m) return;  // whole warp exits together
    const float* a_row = a + static_cast<size_t>(row) * n;
    float sum = 0.0f;
    // Lanes stride the row (coalesced); x is read through the read-only cache.
    for (int j = lane; j < n; j += 32) sum = fmaf(a_row[j], __ldg(&x[j]), sum);
    // Warp reduction; lane 0 writes y[row].
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) y[row] = sum;
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    // 8 rows (warps) per block.
    const int num_blocks = (M + kWarpsPerBlock - 1) / kWarpsPerBlock;
    gemvWarpPerRow<<<num_blocks, kWarpsPerBlock * 32>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}

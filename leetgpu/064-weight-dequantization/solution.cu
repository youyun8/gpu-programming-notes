// Weight Dequantization (LeetGPU)
// https://leetgpu.com/challenges/weight-dequantization
//
// Y[i][j] = X[i][j] * S[i / T][j / T]. A 2D grid with threadIdx.x along the
// columns keeps X/Y accesses coalesced; the scale read is shared by up to
// T x T threads and always hits cache.
#include <cuda_runtime.h>

constexpr int kBlockX = 64;
constexpr int kBlockY = 4;

__global__ void dequantize(const float* x, const float* s, float* y, int m, int n, int tile) {
    // One thread per element: 64 x 4 threads, x along the contiguous columns (coalesced).
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;
    // Each TILE x TILE block of X shares one scale: S[row / TILE][col / TILE].
    const int s_cols = (n + tile - 1) / tile;
    const size_t idx = static_cast<size_t>(row) * n + col;
    y[idx] = x[idx] * s[(row / tile) * s_cols + col / tile];
}

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid((N + kBlockX - 1) / kBlockX, (M + kBlockY - 1) / kBlockY);
    dequantize<<<grid, block>>>(X, S, Y, M, N, TILE_SIZE);
    cudaDeviceSynchronize();
}

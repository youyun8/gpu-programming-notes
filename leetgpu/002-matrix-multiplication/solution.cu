// Matrix Multiplication (LeetGPU)
// https://leetgpu.com/challenges/matrix-multiplication
//
// C (m x k) = A (m x n) * B (n x k), all row-major.
#include <cuda_runtime.h>

constexpr int kTile = 16;

// v1: kept for reference, not launched.
__global__ void matmulNaive(const float* a, const float* b, float* c, int m, int n, int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

// v2: shared-memory tiling.
__global__ void matmulTiled(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float a_tile[kTile][kTile];
    __shared__ float b_tile[kTile][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;

    float sum = 0.0f;
    for (int tile_start = 0; tile_start < n; tile_start += kTile) {
        const int a_col = tile_start + tx;
        const int b_row = tile_start + ty;
        // Out-of-range elements are padded with zeros so every thread
        // still participates in the barriers below.
        a_tile[ty][tx] = (row < m && a_col < n) ? a[row * n + a_col] : 0.0f;
        b_tile[ty][tx] = (b_row < n && col < k) ? b[b_row * k + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int i = 0; i < kTile; ++i) {
            sum += a_tile[ty][i] * b_tile[i][tx];
        }
        __syncthreads();
    }

    if (row < m && col < k) {
        c[row * k + col] = sum;
    }
}

// a, b, c are device pointers
extern "C" void solve(const float* a, const float* b, float* c, int m, int n, int k) {
    const dim3 block(kTile, kTile);
    const dim3 grid((k + kTile - 1) / kTile, (m + kTile - 1) / kTile);
    matmulTiled<<<grid, block>>>(a, b, c, m, n, k);
    cudaDeviceSynchronize();
}

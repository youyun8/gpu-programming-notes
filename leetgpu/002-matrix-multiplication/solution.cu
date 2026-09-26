// Matrix Multiplication (LeetGPU)
// https://leetgpu.com/challenges/matrix-multiplication
//
// C (M x K) = A (M x N) * B (N x K), row-major fp32.
// 64x64 block tile, 16-wide K slices in shared memory, 4x4 outputs per thread.
#include <cuda_runtime.h>

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreads = 256;  // 16 x 16 threads, each owns a 4 x 4 strided sub-tile

__global__ void __launch_bounds__(kThreads)
sgemmTiled(const float* a, const float* b, float* c, int rows, int inner, int cols) {
    // A is stored transposed in shared memory so both operands are read row-wise.
    __shared__ float a_tile[kTileK][kTileM + 4];
    __shared__ float b_tile[kTileK][kTileN + 4];

    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int row0 = blockIdx.y * kTileM;
    const int col0 = blockIdx.x * kTileN;

    float acc[4][4] = {};
    for (int k0 = 0; k0 < inner; k0 += kTileK) {
        for (int i = tid; i < kTileM * kTileK; i += kThreads) {
            const int r = i / kTileK;
            const int kk = i % kTileK;
            const int gr = row0 + r;
            const int gk = k0 + kk;
            a_tile[kk][r] = (gr < rows && gk < inner) ? a[static_cast<size_t>(gr) * inner + gk] : 0.0f;
        }
        for (int i = tid; i < kTileK * kTileN; i += kThreads) {
            const int kk = i / kTileN;
            const int cc = i % kTileN;
            const int gk = k0 + kk;
            const int gc = col0 + cc;
            b_tile[kk][cc] = (gk < inner && gc < cols) ? b[static_cast<size_t>(gk) * cols + gc] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            float a_frag[4];
            float b_frag[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a_frag[i] = a_tile[kk][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b_frag[j] = b_tile[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = row0 + ty + 16 * i;
        if (r >= rows) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + tx + 16 * j;
            if (col < cols) c[static_cast<size_t>(r) * cols + col] = acc[i][j];
        }
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    const dim3 grid((K + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM);
    sgemmTiled<<<grid, kThreads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

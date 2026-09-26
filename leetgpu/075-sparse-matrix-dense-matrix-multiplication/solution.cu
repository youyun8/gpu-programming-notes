// Sparse Matrix-Dense Matrix Multiplication (LeetGPU)
// https://leetgpu.com/challenges/sparse-matrix-dense-matrix-multiplication
//
// A (M x N) is given densely and is only 60-70% zeros. At that density a
// dense, register-blocked SGEMM beats any sparse format on a GPU: CSR needs a
// conversion pass, and its irregular gathers from B cost more than the ~2x
// FLOPs saved. So: the 64 x 64 tiled SGEMM (4 x 4 outputs per thread).
#include <cuda_runtime.h>

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreads = 256;

__global__ void __launch_bounds__(kThreads)
sgemm(const float* a, const float* b, float* c, int rows, int inner, int cols) {
    // The sparse matrix arrives dense, so this is a register-blocked SGEMM:
    // 64 x 64 tile per block, 16 x 16 threads with 4 x 4 outputs each.
    __shared__ float a_tile[kTileK][kTileM + 4];
    __shared__ float b_tile[kTileK][kTileN + 4];
    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int row0 = blockIdx.y * kTileM;
    const int col0 = blockIdx.x * kTileN;
    float acc[4][4] = {};
    // Main loop over K in slices of 16: stage A (transposed) and B, zero outside.
    for (int k0 = 0; k0 < inner; k0 += kTileK) {
        for (int i = tid; i < kTileM * kTileK; i += kThreads) {
            const int r = i / kTileK;
            const int kk = i % kTileK;
            a_tile[kk][r] = (row0 + r < rows && k0 + kk < inner) ? a[static_cast<size_t>(row0 + r) * inner + k0 + kk] : 0.0f;
        }
        for (int i = tid; i < kTileK * kTileN; i += kThreads) {
            const int kk = i / kTileN;
            const int cc = i % kTileN;
            b_tile[kk][cc] = (k0 + kk < inner && col0 + cc < cols) ? b[static_cast<size_t>(k0 + kk) * cols + col0 + cc] : 0.0f;
        }
        // Panels complete before anyone reads them.
        __syncthreads();
        // Outer products: 4 + 4 shared loads feed 16 FMAs.
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
        // Everyone is done with the panels before the next slice overwrites them.
        __syncthreads();
    }
    // Store the 4 x 4 patch in bounds.
    for (int i = 0; i < 4; ++i) {
        const int r = row0 + ty + 16 * i;
        if (r >= rows) continue;
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + tx + 16 * j;
            if (col < cols) c[static_cast<size_t>(r) * cols + col] = acc[i][j];
        }
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K, int nnz) {
    // One block per 64 x 64 tile of C (M x K; the reduction length is N).
    const dim3 grid((K + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM);
    sgemm<<<grid, kThreads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

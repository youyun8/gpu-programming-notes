// FP16 Batched Matrix Multiplication (LeetGPU)
// https://leetgpu.com/challenges/fp16-batched-matrix-multiplication
//
// C[b] = A[b] (M x K) * B[b] (K x N), fp16 in/out, fp32 accumulation on
// tensor cores (WMMA 16x16x16). Same 64 x 64 block tile / 2 x 2 fragments per
// warp as the FP16 GEMM problem, with the batch index on gridDim.z. Shared
// tiles are zero-padded at the edges, so any M, N, K >= 1 works.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 32;
constexpr int kThreads = 128;
constexpr int kPadHalf = 8;
constexpr int kPadFloat = 4;

__global__ void __launch_bounds__(kThreads) hgemmBatched(const half* a, const half* b, half* c, int m, int n, int k) {
    // Shared tiles: A and B slices in half precision and the fp32 C tile for the epilogue
    // (32-byte aligned, padded leading dimensions as WMMA requires).
    __shared__ __align__(32) half a_s[kTileM][kTileK + kPadHalf];
    __shared__ __align__(32) half b_s[kTileK][kTileN + kPadHalf];
    __shared__ __align__(32) float c_s[kTileM][kTileN + kPadFloat];

    // blockIdx.z selects the batch: offset the three matrix pointers to it.
    const size_t batch = blockIdx.z;
    a += batch * m * k;
    b += batch * k * n;
    c += batch * m * n;

    // 4 warps in a 2 x 2 layout; each warp owns a 32 x 32 sub-tile = 2 x 2 WMMA fragments.
    const int warp = threadIdx.x / 32;
    const int warp_row = (warp / 2) * 32;
    const int warp_col = (warp % 2) * 32;
    const int row0 = blockIdx.y * kTileM;
    const int col0 = blockIdx.x * kTileN;

    // fp32 accumulators.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    const half zero = __float2half(0.0f);
    // Main loop over K in slices of 32: stage A and B (zero outside the matrices).
    for (int k0 = 0; k0 < k; k0 += kTileK) {
        for (int i = threadIdx.x; i < kTileM * kTileK; i += kThreads) {
            const int r = i / kTileK;
            const int kk = i % kTileK;
            a_s[r][kk] = (row0 + r < m && k0 + kk < k) ? a[static_cast<size_t>(row0 + r) * k + k0 + kk] : zero;
        }
        for (int i = threadIdx.x; i < kTileK * kTileN; i += kThreads) {
            const int kk = i / kTileN;
            const int cc = i % kTileN;
            b_s[kk][cc] = (k0 + kk < k && col0 + cc < n) ? b[static_cast<size_t>(k0 + kk) * n + col0 + cc] : zero;
        }
        // Tiles complete before the tensor-core loads.
        __syncthreads();
        // Two k-steps of 16: 2 A and 2 B fragments, 4 tensor-core MMAs.
#pragma unroll
        for (int kk = 0; kk < kTileK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a_frag[i], &a_s[warp_row + 16 * i][kk], kTileK + kPadHalf);
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b_frag[j], &b_s[kk][warp_col + 16 * j], kTileN + kPadHalf);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        // Everyone is done with the tiles before the next slice overwrites them.
        __syncthreads();
    }
    // Epilogue: accumulators -> shared fp32 tile -> half, stored in bounds.
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&c_s[warp_row + 16 * i][warp_col + 16 * j], acc[i][j], kTileN + kPadFloat, wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < kTileM * kTileN; i += kThreads) {
        const int r = i / kTileN;
        const int cc = i % kTileN;
        if (row0 + r < m && col0 + cc < n) c[static_cast<size_t>(row0 + r) * n + col0 + cc] = __float2half(c_s[r][cc]);
    }
}

// A, B, C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    // One 128-thread block per 64 x 64 output tile per batch.
    const dim3 grid((N + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM, BATCH);
    hgemmBatched<<<grid, kThreads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

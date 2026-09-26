// GEMM (FP16) (LeetGPU)
// https://leetgpu.com/challenges/general-matrix-multiplication-gemm
//
// C = alpha * A B + beta * C, A: M x K, B: K x N, C: M x N, all fp16 row-major,
// fp32 accumulation on tensor cores via WMMA.
//
// Block = 4 warps computes a 64 x 64 tile of C; each warp owns 32 x 32 = 2 x 2
// WMMA 16x16x16 fragments. A/B slices (64 x 32 and 32 x 64) are staged in shared
// memory with zero padding at the matrix edges, so M, N, K need not be
// multiples of 16. The epilogue goes through shared memory to apply
// alpha/beta and convert to fp16 with bounds checks.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 32;
constexpr int kThreads = 128;
constexpr int kPadHalf = 8;   // keeps ldm a multiple of 8 halves and staggers banks
constexpr int kPadFloat = 4;  // ldm a multiple of 4 floats

__global__ void __launch_bounds__(kThreads)
hgemmWmma(const half* a, const half* b, half* c, int m, int n, int k, float alpha, float beta) {
    __shared__ __align__(32) half a_s[kTileM][kTileK + kPadHalf];
    __shared__ __align__(32) half b_s[kTileK][kTileN + kPadHalf];
    __shared__ __align__(32) float c_s[kTileM][kTileN + kPadFloat];

    const int warp = threadIdx.x / 32;
    const int warp_row = (warp / 2) * 32;
    const int warp_col = (warp % 2) * 32;
    const int row0 = blockIdx.y * kTileM;
    const int col0 = blockIdx.x * kTileN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    const half zero = __float2half(0.0f);
    for (int k0 = 0; k0 < k; k0 += kTileK) {
        for (int i = threadIdx.x; i < kTileM * kTileK; i += kThreads) {
            const int r = i / kTileK;
            const int kk = i % kTileK;
            const int gr = row0 + r;
            const int gk = k0 + kk;
            a_s[r][kk] = (gr < m && gk < k) ? a[static_cast<size_t>(gr) * k + gk] : zero;
        }
        for (int i = threadIdx.x; i < kTileK * kTileN; i += kThreads) {
            const int kk = i / kTileN;
            const int cc = i % kTileN;
            const int gk = k0 + kk;
            const int gc = col0 + cc;
            b_s[kk][cc] = (gk < k && gc < n) ? b[static_cast<size_t>(gk) * n + gc] : zero;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kTileK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a_frag[i], &a_s[warp_row + 16 * i][kk], kTileK + kPadHalf);
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b_frag[j], &b_s[kk][warp_col + 16 * j], kTileN + kPadHalf);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&c_s[warp_row + 16 * i][warp_col + 16 * j], acc[i][j], kTileN + kPadFloat,
                                    wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < kTileM * kTileN; i += kThreads) {
        const int r = i / kTileN;
        const int cc = i % kTileN;
        const int gr = row0 + r;
        const int gc = col0 + cc;
        if (gr < m && gc < n) {
            const size_t idx = static_cast<size_t>(gr) * n + gc;
            const float old = __half2float(c[idx]);
            c[idx] = __float2half(alpha * c_s[r][cc] + beta * old);
        }
    }
}

// A, B, C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    const dim3 grid((N + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM);
    hgemmWmma<<<grid, kThreads>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}

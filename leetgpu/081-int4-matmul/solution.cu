// INT4 MatMul (W4A16) (LeetGPU)
// https://leetgpu.com/challenges/int4-weight-only-quantized-matmul
//
// y (M x N) = x (M x K, fp16) * W^T, W[n][k] = (nibble - 8) * scale[n][k / g].
// Weight-only quantization: W stays packed (0.5 byte/weight) in global memory
// and is dequantized on the fly while a tile is staged into shared memory, so
// the tensor cores (WMMA fp16, fp32 accumulate) see an ordinary fp16 tile.
//   - block = 4 warps, 64 x 64 output tile, 32-wide K slices;
//   - W is N x K row-major, i.e. W^T is K x N column-major: the B fragment is
//     loaded with wmma::col_major straight from the [n][k] shared tile.
#include <cstdint>
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

__global__ void __launch_bounds__(kThreads)
w4a16Gemm(const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int m, int n, int k, int group) {
    __shared__ __align__(32) __half x_s[kTileM][kTileK + kPadHalf];
    __shared__ __align__(32) __half w_s[kTileN][kTileK + kPadHalf];  // [n][k]
    __shared__ __align__(32) float c_s[kTileM][kTileN + kPadFloat];

    const int warp = threadIdx.x / 32;
    const int warp_row = (warp / 2) * 32;
    const int warp_col = (warp % 2) * 32;
    const int row0 = blockIdx.y * kTileM;
    const int col0 = blockIdx.x * kTileN;
    const int num_groups = k / group;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    const __half zero = __float2half(0.0f);
    for (int k0 = 0; k0 < k; k0 += kTileK) {
        for (int i = threadIdx.x; i < kTileM * kTileK; i += kThreads) {
            const int r = i / kTileK;
            const int kk = i % kTileK;
            x_s[r][kk] = (row0 + r < m && k0 + kk < k) ? x[static_cast<size_t>(row0 + r) * k + k0 + kk] : zero;
        }
        // Each thread unpacks one byte = two consecutive weights of one row of W.
        for (int i = threadIdx.x; i < kTileN * (kTileK / 2); i += kThreads) {
            const int nn = i / (kTileK / 2);
            const int kk = (i % (kTileK / 2)) * 2;
            const int gn = col0 + nn;
            const int gk = k0 + kk;
            __half w0 = zero, w1 = zero;
            if (gn < n && gk < k) {
                const uint8_t byte = w_q[static_cast<size_t>(gn) * (k / 2) + gk / 2];
                const float s0 = __half2float(scales[static_cast<size_t>(gn) * num_groups + gk / group]);
                const float s1 = __half2float(scales[static_cast<size_t>(gn) * num_groups + (gk + 1) / group]);
                w0 = __float2half(static_cast<float>(static_cast<int>(byte >> 4) - 8) * s0);
                w1 = __float2half(static_cast<float>(static_cast<int>(byte & 0xF) - 8) * s1);
            }
            w_s[nn][kk] = w0;
            w_s[nn][kk + 1] = w1;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kTileK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag[2];
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a_frag[i], &x_s[warp_row + 16 * i][kk], kTileK + kPadHalf);
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b_frag[j], &w_s[warp_col + 16 * j][kk], kTileK + kPadHalf);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&c_s[warp_row + 16 * i][warp_col + 16 * j], acc[i][j], kTileN + kPadFloat, wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < kTileM * kTileN; i += kThreads) {
        const int r = i / kTileN;
        const int cc = i % kTileN;
        if (row0 + r < m && col0 + cc < n) y[static_cast<size_t>(row0 + r) * n + col0 + cc] = __float2half(c_s[r][cc]);
    }
}

// x, w_q, scales, y are device pointers
extern "C" void solve(const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M, int N, int K,
                      int group_size) {
    const dim3 grid((N + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM);
    w4a16Gemm<<<grid, kThreads>>>(x, w_q, scales, y, M, N, K, group_size);
    cudaDeviceSynchronize();
}

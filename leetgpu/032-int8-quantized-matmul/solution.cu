// INT8 Quantized MatMul (LeetGPU)
// https://leetgpu.com/challenges/int8-quantized-matmul
//
// C_q = clamp(round(sum_k (A-zA)(B-zB) * sA*sB/sC) + zC, -128, 127)
//
// (A - zA) does not fit in int8, so the zero points are folded out
// algebraically and the raw int8 product runs on tensor cores (WMMA s8 -> s32):
//   sum (A-zA)(B-zB) = A.B - zB*rowsum(A) - zA*colsum(B) + K*zA*zB
// Row sums of A and column sums of B come from two tiny pre-pass kernels.
// Shared tiles are stored as contiguous 16 x 16 blocks: WMMA needs 32-byte
// aligned pointers, and with 1-byte elements a 16-column offset inside a
// padded row would only be 16-byte aligned.
// The requantization repeats the reference's float32 operation order exactly
// ((acc * sA) * sB / sC, round-half-to-even) because the check is exact.
#include <cstdint>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 32;
constexpr int kThreads = 128;
constexpr int kFrag = 16;

__global__ void rowSums(const int8_t* a, int* row_sum, int m, int k) {
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= m) return;
    int s = 0;
    for (int j = lane; j < k; j += 32) s += a[static_cast<size_t>(row) * k + j];
    for (int offset = 16; offset > 0; offset >>= 1) s += __shfl_down_sync(0xffffffffu, s, offset);
    if (lane == 0) row_sum[row] = s;
}

__global__ void colSums(const int8_t* b, int* col_sum, int k, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    int s = 0;
    for (int i = 0; i < k; ++i) s += b[static_cast<size_t>(i) * n + col];
    col_sum[col] = s;
}

__global__ void __launch_bounds__(kThreads)
imma(const int8_t* a, const int8_t* b, int8_t* c, const int* row_sum, const int* col_sum, int m, int n, int k,
     float scale_a, float scale_b, float scale_c, int za, int zb, int zc) {
    __shared__ __align__(32) int8_t a_s[kTileM / kFrag][kTileK / kFrag][kFrag][kFrag];
    __shared__ __align__(32) int8_t b_s[kTileK / kFrag][kTileN / kFrag][kFrag][kFrag];
    __shared__ __align__(32) int c_s[kTileM][kTileN + 8];

    const int warp = threadIdx.x / 32;
    const int warp_row = (warp / 2) * 32;
    const int warp_col = (warp % 2) * 32;
    const int row0 = blockIdx.y * kTileM;
    const int col0 = blockIdx.x * kTileN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0);

    for (int k0 = 0; k0 < k; k0 += kTileK) {
        for (int i = threadIdx.x; i < kTileM * kTileK; i += kThreads) {
            const int r = i / kTileK;
            const int kk = i % kTileK;
            a_s[r / kFrag][kk / kFrag][r % kFrag][kk % kFrag] =
                (row0 + r < m && k0 + kk < k) ? a[static_cast<size_t>(row0 + r) * k + k0 + kk] : int8_t(0);
        }
        for (int i = threadIdx.x; i < kTileK * kTileN; i += kThreads) {
            const int kk = i / kTileN;
            const int cc = i % kTileN;
            b_s[kk / kFrag][cc / kFrag][kk % kFrag][cc % kFrag] =
                (k0 + kk < k && col0 + cc < n) ? b[static_cast<size_t>(k0 + kk) * n + col0 + cc] : int8_t(0);
        }
        __syncthreads();
#pragma unroll
        for (int kb = 0; kb < kTileK / kFrag; ++kb) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major> b_frag[2];
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(a_frag[i], reinterpret_cast<const signed char*>(&a_s[warp_row / kFrag + i][kb][0][0]), kFrag);
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(b_frag[j], reinterpret_cast<const signed char*>(&b_s[kb][warp_col / kFrag + j][0][0]), kFrag);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&c_s[warp_row + 16 * i][warp_col + 16 * j], acc[i][j], kTileN + 8, wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < kTileM * kTileN; i += kThreads) {
        const int r = i / kTileN;
        const int cc = i % kTileN;
        const int gr = row0 + r;
        const int gc = col0 + cc;
        if (gr >= m || gc >= n) continue;
        const int acc_true = c_s[r][cc] - zb * row_sum[gr] - za * col_sum[gc] + k * za * zb;
        float v = static_cast<float>(acc_true) * scale_a;
        v = v * scale_b;
        v = v / scale_c;
        int q = static_cast<int>(rintf(v)) + zc;
        q = q < -128 ? -128 : (q > 127 ? 127 : q);
        c[static_cast<size_t>(gr) * n + gc] = static_cast<int8_t>(q);
    }
}

// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, float scale_A, float scale_B,
                      float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {
    int* sums = nullptr;
    cudaMalloc(&sums, (static_cast<size_t>(M) + N) * sizeof(int));
    int* row_sum = sums;
    int* col_sum = sums + M;
    rowSums<<<(M + 7) / 8, 256>>>(A, row_sum, M, K);
    colSums<<<(N + 255) / 256, 256>>>(B, col_sum, K, N);
    const dim3 grid((N + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM);
    imma<<<grid, kThreads>>>(A, B, C, row_sum, col_sum, M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B,
                             zero_point_C);
    cudaDeviceSynchronize();
    cudaFree(sums);
}

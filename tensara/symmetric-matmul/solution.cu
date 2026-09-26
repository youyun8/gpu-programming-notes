// Symmetric Matrix Multiplication (Tensara)
// https://tensara.org/problems/symmetric-matmul
//
// C = A * B for symmetric N x N matrices. Symmetry does not reduce the work of a
// dense product; it only means B^T = B, so either layout can be used. The
// register-blocked SGEMM is used as is.
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Building blocks shared by the transformer-block kernels in this file.
// ---------------------------------------------------------------------------
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kGemmThreads = 256;

// C[r][c] = epi(sum_k A[r][k] * B'[k][c], r, c)
//   A: rows x inner, row stride lda.
//   B' = B^T (kTransB, B stored cols x inner, "NT") or B (inner x cols, "NN"), row stride ldb.
// 64 x 64 block tile, 16-wide K slices in shared memory, 4 x 4 outputs per thread.
template <bool kTransB, class Epi>
__global__ void __launch_bounds__(kGemmThreads)
gemmKernel(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int rows, int inner, int cols, Epi epi) {
    __shared__ float a_tile[kTileK][kTileM + 4];
    __shared__ float b_tile[kTileK][kTileN + 4];
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int row0 = blockIdx.y * kTileM, col0 = blockIdx.x * kTileN;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < inner; k0 += kTileK) {
        for (int i = tid; i < kTileM * kTileK; i += kGemmThreads) {
            const int r = i / kTileK, kk = i % kTileK;
            a_tile[kk][r] = (row0 + r < rows && k0 + kk < inner) ? a[static_cast<size_t>(row0 + r) * lda + k0 + kk] : 0.0f;
        }
        for (int i = tid; i < kTileK * kTileN; i += kGemmThreads) {
            if (kTransB) {
                const int cc = i / kTileK, kk = i % kTileK;
                b_tile[kk][cc] = (col0 + cc < cols && k0 + kk < inner) ? b[static_cast<size_t>(col0 + cc) * ldb + k0 + kk] : 0.0f;
            } else {
                const int kk = i / kTileN, cc = i % kTileN;
                b_tile[kk][cc] = (k0 + kk < inner && col0 + cc < cols) ? b[static_cast<size_t>(k0 + kk) * ldb + col0 + cc] : 0.0f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            float af[4], bf[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) af[i] = a_tile[kk][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) bf[j] = b_tile[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i) {
        const int r = row0 + ty + 16 * i;
        if (r >= rows) continue;
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + tx + 16 * j;
            if (col < cols) c[static_cast<size_t>(r) * ldc + col] = epi(acc[i][j], r, col);
        }
    }
}

template <bool kTransB, class Epi>
static void gemm(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int rows, int inner, int cols, Epi epi) {
    const dim3 grid((cols + kTileN - 1) / kTileN, (rows + kTileM - 1) / kTileM);
    gemmKernel<kTransB, Epi><<<grid, kGemmThreads>>>(a, lda, b, ldb, c, ldc, rows, inner, cols, epi);
}

struct NoEpi {
    __device__ float operator()(float v, int, int) const { return v; }
};
struct BiasEpi {
    const float* bias;
    __device__ float operator()(float v, int, int c) const { return v + bias[c]; }
};

// input_a, input_b, output_c are device pointers
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n) {
    const int ni = static_cast<int>(n);
    gemm<false>(input_a, ni, input_b, ni, output_c, ni, ni, ni, ni, NoEpi{});
}

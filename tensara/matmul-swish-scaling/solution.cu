// Matmul + Swish + Scaling (Tensara)
// https://tensara.org/problems/matmul-swish-scaling
//
// out = scale * swish(A B), with swish and scaling fused into the GEMM epilogue.
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
    // Shared-memory staging buffers for one K-slice. A is stored transposed
    // ([k][m]) so that both operands are read along rows; +4 padding avoids bank conflicts.
    __shared__ float a_tile[kTileK][kTileM + 4];
    __shared__ float b_tile[kTileK][kTileN + 4];
    // Thread coordinates: a 16 x 16 grid of threads, each owning a 4 x 4 patch of
    // C at rows ty + 16i and columns tx + 16j (stride 16 keeps stores coalesced).
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    // Top-left corner of this block's 64 x 64 output tile.
    const int row0 = blockIdx.y * kTileM, col0 = blockIdx.x * kTileN;
    // Per-thread accumulators, kept in registers (fully unrolled loops below).
    float acc[4][4] = {};
    // Main loop over the reduction dimension, 16 columns of A / rows of B at a time.
    for (int k0 = 0; k0 < inner; k0 += kTileK) {
        // Stage the 64 x 16 panel of A (zero-filled outside the matrix), transposing it.
        for (int i = tid; i < kTileM * kTileK; i += kGemmThreads) {
            const int r = i / kTileK, kk = i % kTileK;
            a_tile[kk][r] = (row0 + r < rows && k0 + kk < inner) ? a[static_cast<size_t>(row0 + r) * lda + k0 + kk] : 0.0f;
        }
        // Stage the 16 x 64 panel of B. With kTransB, B is stored (cols x inner) and is
        // transposed while loading; otherwise it is copied row by row (coalesced).
        for (int i = tid; i < kTileK * kTileN; i += kGemmThreads) {
            if (kTransB) {
                const int cc = i / kTileK, kk = i % kTileK;
                b_tile[kk][cc] = (col0 + cc < cols && k0 + kk < inner) ? b[static_cast<size_t>(col0 + cc) * ldb + k0 + kk] : 0.0f;
            } else {
                const int kk = i / kTileN, cc = i % kTileN;
                b_tile[kk][cc] = (k0 + kk < inner && col0 + cc < cols) ? b[static_cast<size_t>(k0 + kk) * ldb + col0 + cc] : 0.0f;
            }
        }
        // Both panels must be complete before any thread reads them.
        __syncthreads();
        // Inner product: per k, load 4 values of A and 4 of B into registers and do a
        // 4 x 4 outer product (16 FMAs for 8 shared loads).
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
        // Wait until every thread is done with the panels before the next slice overwrites them.
        __syncthreads();
    }
    // Epilogue: apply the fused operation (bias, activation, ...) and store in bounds.
    for (int i = 0; i < 4; ++i) {
        const int r = row0 + ty + 16 * i;
        if (r >= rows) continue;
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + tx + 16 * j;
            if (col < cols) c[static_cast<size_t>(r) * ldc + col] = epi(acc[i][j], r, col);
        }
    }
}

// Host helper: one block per 64 x 64 output tile.
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

struct SwishScaleEpi {
    float scale;
    __device__ float operator()(float v, int, int) const { return scale * (v / (1.0f + expf(-v))); }
};

// A, B, output are device pointers
extern "C" void solution(const float* A, const float* B, float scale, float* output, size_t M, size_t N, size_t K) {
    gemm<false>(A, static_cast<int>(K), B, static_cast<int>(N), output, static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                static_cast<int>(N), SwishScaleEpi{scale});
}

// GEMM technique 1: 128-bit (float4) global and shared-memory accesses.
//
// C = A B, FP32, row-major. A 128 x 128 block tile, K slices of 8, 256 threads,
// 8 x 8 outputs per thread. Every global load, every fragment load from shared
// memory and every store of C is 16 bytes wide:
//
//   global -> registers : one float4 of A and one float4 of B per thread (LDG.E.128)
//   registers -> shared : A is stored transposed (4 scalar STS), B as one float4 (STS.128)
//   shared -> registers : 2 + 2 float4 fragment loads per k step (LDS.128)
//   registers -> global : 16 float4 stores of C per thread (STG.E.128)
//
// Each thread owns rows {4ty..4ty+3} and {64+4ty..64+4ty+3}, and the same split of
// columns with tx, so the fragment loads of 8 consecutive lanes cover 128 contiguous
// bytes (conflict-free) and the stores of C are coalesced.
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 01-vectorized.cu -o vectorized && ./vectorized
#include "harness.cuh"

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 8;
constexpr int kThreads = 256;
constexpr int kPadA = 4;  // keeps the transposed stores of A conflict-free and rows 16-byte aligned

// Loads 4 consecutive floats of row `row` starting at column `col` (col % 4 == 0),
// with zeros outside the matrix. kVec: the row length is a multiple of 4, so the
// 4 floats are 16-byte aligned and either all inside or all outside.
template <bool kVec>
__device__ __forceinline__ float4 load4(const float* p, int rows, int cols, int row, int col) {
    if (row >= rows) return make_float4(0.f, 0.f, 0.f, 0.f);
    const float* src = p + static_cast<size_t>(row) * cols + col;
    if (kVec) return col < cols ? *reinterpret_cast<const float4*>(src) : make_float4(0.f, 0.f, 0.f, 0.f);
    return make_float4(col < cols ? src[0] : 0.f, col + 1 < cols ? src[1] : 0.f, col + 2 < cols ? src[2] : 0.f,
                       col + 3 < cols ? src[3] : 0.f);
}

template <bool kVec>
__device__ __forceinline__ void store4(float* p, int rows, int cols, int row, int col, float4 v) {
    if (row >= rows) return;
    float* dst = p + static_cast<size_t>(row) * cols + col;
    if (kVec) {
        if (col < cols) *reinterpret_cast<float4*>(dst) = v;
        return;
    }
    if (col < cols) dst[0] = v.x;
    if (col + 1 < cols) dst[1] = v.y;
    if (col + 2 < cols) dst[2] = v.z;
    if (col + 3 < cols) dst[3] = v.w;
}

template <bool kVec>
__global__ void __launch_bounds__(kThreads) sgemmVectorized(const float* __restrict__ a, const float* __restrict__ b,
                                                            float* __restrict__ c, int m, int n, int k) {
    __shared__ __align__(16) float a_s[kBlockK][kBlockM + kPadA];  // transposed: a_s[kk][row]
    __shared__ __align__(16) float b_s[kBlockK][kBlockN];

    const int tid = threadIdx.x;
    const int tx = tid % 16;  // column group of the thread
    const int ty = tid / 16;  // row group of the thread
    const int row0 = blockIdx.y * kBlockM;
    const int col0 = blockIdx.x * kBlockN;

    // Which float4 each thread moves from global memory: A is 128 x 8 (2 float4 per row),
    // B is 8 x 128 (32 float4 per row). 256 threads move exactly one of each.
    const int a_row = tid / 2, a_col = (tid % 2) * 4;
    const int b_row = tid / 32, b_col = (tid % 32) * 4;

    float acc[8][8] = {};

    for (int k0 = 0; k0 < k; k0 += kBlockK) {
        const float4 av = load4<kVec>(a, m, k, row0 + a_row, k0 + a_col);
        const float4 bv = load4<kVec>(b, k, n, k0 + b_row, col0 + b_col);
        // A goes in transposed, so that a thread's 4 rows are 4 consecutive words.
        a_s[a_col + 0][a_row] = av.x;
        a_s[a_col + 1][a_row] = av.y;
        a_s[a_col + 2][a_row] = av.z;
        a_s[a_col + 3][a_row] = av.w;
        *reinterpret_cast<float4*>(&b_s[b_row][b_col]) = bv;
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kBlockK; ++kk) {
            // Four LDS.128: rows 4ty.. and 64+4ty.. of A, columns 4tx.. and 64+4tx.. of B.
            const float4 a_lo = *reinterpret_cast<const float4*>(&a_s[kk][4 * ty]);
            const float4 a_hi = *reinterpret_cast<const float4*>(&a_s[kk][64 + 4 * ty]);
            const float4 b_lo = *reinterpret_cast<const float4*>(&b_s[kk][4 * tx]);
            const float4 b_hi = *reinterpret_cast<const float4*>(&b_s[kk][64 + 4 * tx]);
            const float a_frag[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w, a_hi.x, a_hi.y, a_hi.z, a_hi.w};
            const float b_frag[8] = {b_lo.x, b_lo.y, b_lo.z, b_lo.w, b_hi.x, b_hi.y, b_hi.z, b_hi.w};
            // 64 FMAs: the outer product of the two fragments.
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();  // the next slice overwrites the tiles
    }

    // Epilogue: two float4 per output row; lanes with consecutive tx write consecutive 16 bytes.
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row0 + (i < 4 ? 4 * ty + i : 64 + 4 * ty + (i - 4));
        store4<kVec>(c, m, n, row, col0 + 4 * tx, make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]));
        store4<kVec>(c, m, n, row, col0 + 64 + 4 * tx, make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]));
    }
}

void launchVectorized(const float* a, const float* b, float* c, int m, int n, int k) {
    const dim3 grid(gemm::ceilDiv(n, kBlockN), gemm::ceilDiv(m, kBlockM));
    // float4 accesses need every row to start on a 16-byte boundary.
    if (k % 4 == 0 && n % 4 == 0)
        sgemmVectorized<true><<<grid, kThreads>>>(a, b, c, m, n, k);
    else
        sgemmVectorized<false><<<grid, kThreads>>>(a, b, c, m, n, k);
}

int main(int argc, char** argv) {
    return gemm::runMain<float>("01-vectorized", launchVectorized, argc, argv);
}

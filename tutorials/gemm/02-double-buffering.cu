// GEMM technique 2: double buffering (software pipelining over K).
//
// Same tile shapes and thread mapping as 01-vectorized.cu, but the shared-memory
// tiles exist twice. While the block computes slice s from buffer s % 2, each
// thread already holds slice s + 1 in registers (its global loads were issued
// before the math), and writes it into the other buffer afterwards:
//
//   prologue:  load slice 0 -> registers -> buffer 0; barrier
//   step s:    issue global loads of slice s+1 into registers   (latency starts)
//              64 FMAs x 8 on buffer s % 2                       (latency hidden)
//              registers -> buffer (s+1) % 2; barrier            (one barrier per slice)
//
// Buffer (s+1) % 2 was last read during step s-1, and step s-1 ended with a
// barrier, so overwriting it during step s is safe: one __syncthreads() per slice
// instead of two.
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 02-double-buffering.cu -o double_buffering
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
__global__ void __launch_bounds__(kThreads) sgemmDoubleBuffered(const float* __restrict__ a,
                                                                const float* __restrict__ b,
                                                                float* __restrict__ c, int m, int n, int k) {
    __shared__ __align__(16) float a_s[2][kBlockK][kBlockM + kPadA];
    __shared__ __align__(16) float b_s[2][kBlockK][kBlockN];

    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int row0 = blockIdx.y * kBlockM;
    const int col0 = blockIdx.x * kBlockN;
    const int a_row = tid / 2, a_col = (tid % 2) * 4;
    const int b_row = tid / 32, b_col = (tid % 32) * 4;

    // Registers -> shared memory for one slice (A transposed, as in 01).
    auto storeSlice = [&](int buf, float4 av, float4 bv) {
        a_s[buf][a_col + 0][a_row] = av.x;
        a_s[buf][a_col + 1][a_row] = av.y;
        a_s[buf][a_col + 2][a_row] = av.z;
        a_s[buf][a_col + 3][a_row] = av.w;
        *reinterpret_cast<float4*>(&b_s[buf][b_row][b_col]) = bv;
    };

    float acc[8][8] = {};
    const int num_slices = (k + kBlockK - 1) / kBlockK;

    // Prologue: slice 0 into buffer 0.
    storeSlice(0, load4<kVec>(a, m, k, row0 + a_row, a_col), load4<kVec>(b, k, n, b_row, col0 + b_col));
    __syncthreads();

    for (int s = 0; s < num_slices; ++s) {
        const int buf = s % 2;
        const bool has_next = s + 1 < num_slices;
        // 1. Issue the global loads of the next slice. Nothing uses them until after
        //    the math below, so the warp does not stall on them here.
        float4 a_next = make_float4(0.f, 0.f, 0.f, 0.f), b_next = a_next;
        if (has_next) {
            const int k1 = (s + 1) * kBlockK;
            a_next = load4<kVec>(a, m, k, row0 + a_row, k1 + a_col);
            b_next = load4<kVec>(b, k, n, k1 + b_row, col0 + b_col);
        }
        // 2. Compute on the current buffer.
#pragma unroll
        for (int kk = 0; kk < kBlockK; ++kk) {
            const float4 a_lo = *reinterpret_cast<const float4*>(&a_s[buf][kk][4 * ty]);
            const float4 a_hi = *reinterpret_cast<const float4*>(&a_s[buf][kk][64 + 4 * ty]);
            const float4 b_lo = *reinterpret_cast<const float4*>(&b_s[buf][kk][4 * tx]);
            const float4 b_hi = *reinterpret_cast<const float4*>(&b_s[buf][kk][64 + 4 * tx]);
            const float a_frag[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w, a_hi.x, a_hi.y, a_hi.z, a_hi.w};
            const float b_frag[8] = {b_lo.x, b_lo.y, b_lo.z, b_lo.w, b_hi.x, b_hi.y, b_hi.z, b_hi.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
        }
        // 3. Fill the other buffer, then one barrier: it publishes slice s+1 and also
        //    guarantees that everyone is done reading buffer `buf` before step s+1
        //    overwrites it with slice s+2.
        if (has_next) storeSlice(buf ^ 1, a_next, b_next);
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row0 + (i < 4 ? 4 * ty + i : 64 + 4 * ty + (i - 4));
        store4<kVec>(c, m, n, row, col0 + 4 * tx, make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]));
        store4<kVec>(c, m, n, row, col0 + 64 + 4 * tx, make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]));
    }
}

void launchDoubleBuffered(const float* a, const float* b, float* c, int m, int n, int k) {
    const dim3 grid(gemm::ceilDiv(n, kBlockN), gemm::ceilDiv(m, kBlockM));
    if (k % 4 == 0 && n % 4 == 0)
        sgemmDoubleBuffered<true><<<grid, kThreads>>>(a, b, c, m, n, k);
    else
        sgemmDoubleBuffered<false><<<grid, kThreads>>>(a, b, c, m, n, k);
}

int main(int argc, char** argv) {
    return gemm::runMain<float>("02-double-buffering", launchDoubleBuffered, argc, argv);
}

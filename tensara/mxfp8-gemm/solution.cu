// MXFP8 GEMM (Tensara)
// https://tensara.org/problems/mxfp8-gemm
//
// C (m x n, fp32) = A B^T with MXFP8 operands (A: m x k, B: n x k) and E8M0
// scales per 32 elements in the swizzled 128x4 layout (as produced by
// TorchAO to_mx(..., is_swizzled_scales=True)).
#include <cstdint>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Low-precision formats, decoded / encoded with integer bit manipulation so the
// code does not depend on cuda_fp8.h / cuda_fp4.h or a specific architecture.
//   E2M1 (FP4): 1 sign, 2 exponent, 1 mantissa bit -> {0, .5, 1, 1.5, 2, 3, 4, 6};
//               two values per byte, element 2i in the LOW nibble.
//   E4M3 (FP8): bias 7, max 448, 0x7F / 0xFF are NaN, subnormals below 2^-6.
//   E8M0:       pure power of two 2^(b - 127), 0xFF is NaN (MX block scales).
// Block scales for GEMMs use the 128 x 4 "32_4_4" swizzled layout of
// cuBLAS / CUTLASS block-scaled MMA: scales are grouped in 128-row x 4-column
// atoms of 512 bytes, row r of an atom stored at (r % 32) * 16 + (r / 32) * 4.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float e2m1ToFloat(unsigned int code) {
    const unsigned int mag = code & 7u;
    // codes 0..3 -> 0, .5, 1, 1.5 (m * 0.5); codes 4..7 -> 2, 3, 4, 6 ((2 + (m & 1)) * 2^((m >> 1) - 2))
    const float v = mag < 4u ? 0.5f * static_cast<float>(mag)
                             : static_cast<float>(2u + (mag & 1u)) * static_cast<float>(1u << ((mag >> 1) - 2u));
    return (code & 8u) ? -v : v;
}

__device__ __forceinline__ float e4m3ToFloat(unsigned int b) {
    const unsigned int e = (b >> 3) & 0xFu, m = b & 7u;
    float v;
    if (e == 15u && m == 7u) v = __int_as_float(0x7fc00000);            // NaN
    else if (e == 0u) v = static_cast<float>(m) * 0.001953125f;          // m/8 * 2^-6
    else v = ldexpf(1.0f + static_cast<float>(m) * 0.125f, static_cast<int>(e) - 7);
    return (b & 0x80u) ? -v : v;
}

__device__ __forceinline__ float e8m0ToFloat(unsigned int b) {
    return b == 255u ? __int_as_float(0x7fc00000) : ldexpf(1.0f, static_cast<int>(b) - 127);
}

// Round to nearest even, saturate to +-448 (the "satfinite" conversion).
__device__ __forceinline__ unsigned int floatToE4M3(float x) {
    const unsigned int sign = (__float_as_uint(x) >> 24) & 0x80u;
    const float a = fabsf(x);
    if (a != a) return sign | 0x7Fu;
    if (a >= 464.0f) return sign | 0x7Eu;                      // rounds past 448 -> saturate
    if (a < 0.015625f) {                                         // subnormal range, step 2^-9
        const unsigned int m = static_cast<unsigned int>(rintf(a * 512.0f));
        return sign | m;                                         // m == 8 is exactly the min normal 0x08
    }
    int e;
    const float frac = frexpf(a, &e);                            // a = frac * 2^e, frac in [0.5, 1)
    unsigned int m = static_cast<unsigned int>(rintf((frac * 2.0f - 1.0f) * 8.0f));
    int exp_field = e - 1 + 7;
    if (m == 8u) {
        m = 0u;
        ++exp_field;
    }
    unsigned int code = (static_cast<unsigned int>(exp_field) << 3) | m;
    if (code > 0x7Eu) code = 0x7Eu;
    return sign | code;
}

// Round to nearest, ties to even code, saturate at 6. Midpoints between the
// representable magnitudes are .25 .75 1.25 1.75 2.5 3.5 5 ('>=' where the
// upper neighbour has the even code).
__device__ __forceinline__ unsigned int floatToE2M1(float x) {
    const float a = fabsf(x);
    unsigned int code = 0u;
    code = a > 0.25f ? 1u : code;
    code = a >= 0.75f ? 2u : code;
    code = a > 1.25f ? 3u : code;
    code = a >= 1.75f ? 4u : code;
    code = a > 2.5f ? 5u : code;
    code = a >= 3.5f ? 6u : code;
    code = a > 5.0f ? 7u : code;
    return code | (x < 0.0f ? 8u : 0u);
}

__device__ __forceinline__ size_t swizzledScaleIndex(size_t r, size_t c, size_t cols) {
    const size_t col_blocks = (cols + 3) / 4;
    const size_t ri = r % 128;
    return ((r / 128) * col_blocks + c / 4) * 512 + (ri % 32) * 16 + (ri / 32) * 4 + c % 4;
}

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 32;
constexpr int kThreads = 256;

// C = A * B^T with A (m x k) and B (n x k) stored as quantized rows.
// Each 64 x 64 output tile walks K in 32-wide slices; the slice of A and of B is
// DEQUANTIZED while it is staged into shared memory (fp32, A transposed), so
// the inner product is the ordinary register-blocked fp32 SGEMM (4 x 4 outputs
// per thread). Quantized data is read once per tile, i.e. 4-8x less DRAM
// traffic than a pre-dequantized fp32 GEMM.
__global__ void __launch_bounds__(kThreads)
blockScaledGemm(const uint8_t* __restrict__ qa, const uint8_t* __restrict__ sa, const uint8_t* __restrict__ qb,
                const uint8_t* __restrict__ sb, float* __restrict__ c, int m, int n, int k, float global_scale) {
    __shared__ float a_tile[kTileK][kTileM + 4];
    __shared__ float b_tile[kTileK][kTileN + 4];
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int row0 = blockIdx.y * kTileM, col0 = blockIdx.x * kTileN;
    const size_t scale_cols = static_cast<size_t>(k) / 32;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < k; k0 += kTileK) {
        for (int i = tid; i < kTileM * kTileK; i += kThreads) {
            const int r = i / kTileK, kk = i % kTileK;
            const int gr = row0 + r, gk = k0 + kk;
            a_tile[kk][r] = gr < m ? e4m3ToFloat(qa[static_cast<size_t>(gr) * k + (gk)]) * e8m0ToFloat(sa[swizzledScaleIndex(gr, gk / 32, scale_cols)]) : 0.0f;
        }
        for (int i = tid; i < kTileN * kTileK; i += kThreads) {
            const int cc = i / kTileK, kk = i % kTileK;
            const int gc = col0 + cc, gk = k0 + kk;
            b_tile[kk][cc] = gc < n ? e4m3ToFloat(qb[static_cast<size_t>(gc) * k + (gk)]) * e8m0ToFloat(sb[swizzledScaleIndex(gc, gk / 32, scale_cols)]) : 0.0f;
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
        if (r >= m) continue;
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + tx + 16 * j;
            if (col < n) c[static_cast<size_t>(r) * n + col] = (acc[i][j] * global_scale);
        }
    }
}

// q_a, scale_a, q_b, scale_b, c are device pointers
extern "C" void solution(const uint8_t* q_a, const uint8_t* scale_a, const uint8_t* q_b, const uint8_t* scale_b, float* c, size_t m, size_t n, size_t k) {
    const dim3 grid(static_cast<unsigned>((n + kTileN - 1) / kTileN), static_cast<unsigned>((m + kTileM - 1) / kTileM));
    blockScaledGemm<<<grid, kThreads>>>(q_a, scale_a, q_b, scale_b, c, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1.0f);
}

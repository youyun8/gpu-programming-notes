// NVFP4 GEMM (Tensara)
// https://tensara.org/problems/nvfp4-gemm
//
// C (m x n, fp16) = (A_q * s_a / g_a) (B_q * s_b / g_b)^T: FP4 (E2M1) elements with FP8
// (E4M3) scales per 16 elements in the swizzled 128x4 layout, plus one fp32
// global scale per operand (applied once in the epilogue).
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
    // Low 3 bits: magnitude code; bit 3: sign.
    const unsigned int mag = code & 7u;
    // codes 0..3 -> 0, .5, 1, 1.5 (m * 0.5); codes 4..7 -> 2, 3, 4, 6 ((2 + (m & 1)) * 2^((m >> 1) - 2))
    const float v = mag < 4u ? 0.5f * static_cast<float>(mag)
                             : static_cast<float>(2u + (mag & 1u)) * static_cast<float>(1u << ((mag >> 1) - 2u));
    return (code & 8u) ? -v : v;
}

__device__ __forceinline__ float e4m3ToFloat(unsigned int b) {
    // Split the byte into the 4-bit exponent and 3-bit mantissa fields (bias 7).
    const unsigned int e = (b >> 3) & 0xFu, m = b & 7u;
    float v;
    if (e == 15u && m == 7u) v = __int_as_float(0x7fc00000);            // NaN
    else if (e == 0u) v = static_cast<float>(m) * 0.001953125f;          // m/8 * 2^-6
    else v = ldexpf(1.0f + static_cast<float>(m) * 0.125f, static_cast<int>(e) - 7);
    // Apply the sign bit.
    return (b & 0x80u) ? -v : v;
}

__device__ __forceinline__ float e8m0ToFloat(unsigned int b) {
    return b == 255u ? __int_as_float(0x7fc00000) : ldexpf(1.0f, static_cast<int>(b) - 127);
}

// Round to nearest even, saturate to +-448 (the "satfinite" conversion).
__device__ __forceinline__ unsigned int floatToE4M3(float x) {
    // Keep the sign bit, then encode the magnitude.
    const unsigned int sign = (__float_as_uint(x) >> 24) & 0x80u;
    const float a = fabsf(x);
    // NaN in, NaN out; values that would round past 448 saturate to the largest finite code.
    if (a != a) return sign | 0x7Fu;
    if (a >= 464.0f) return sign | 0x7Eu;                      // rounds past 448 -> saturate
    if (a < 0.015625f) {                                         // subnormal range, step 2^-9
        const unsigned int m = static_cast<unsigned int>(rintf(a * 512.0f));
        return sign | m;                                         // m == 8 is exactly the min normal 0x08
    }
    // Normal range: split a into mantissa and exponent, round the mantissa to 3 bits.
    int e;
    const float frac = frexpf(a, &e);                            // a = frac * 2^e, frac in [0.5, 1)
    unsigned int m = static_cast<unsigned int>(rintf((frac * 2.0f - 1.0f) * 8.0f));
    int exp_field = e - 1 + 7;
    // Mantissa rounded up to 2.0: carry into the exponent.
    if (m == 8u) {
        m = 0u;
        ++exp_field;
    }
    // Assemble the code and clamp to 0x7E (448); 0x7F is NaN.
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
    // Compare against the midpoints between neighbouring E2M1 values
    // {0, .5, 1, 1.5, 2, 3, 4, 6}; ">" vs ">=" picks the even code on ties.
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
    // 128 x 4 atoms of 512 bytes; inside an atom, rows r, r+32, r+64, r+96 are interleaved.
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
                const uint8_t* __restrict__ sb, __half* __restrict__ c, int m, int n, int k, float global_scale) {
    __shared__ float a_tile[kTileK][kTileM + 4];
    __shared__ float b_tile[kTileK][kTileN + 4];
    // 16 x 16 threads, each owning a 4 x 4 patch of C (rows ty + 16i, columns tx + 16j).
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int row0 = blockIdx.y * kTileM, col0 = blockIdx.x * kTileN;
    // One E4M3 scale per 16 elements along K.
    const size_t scale_cols = static_cast<size_t>(k) / 16;
    float acc[4][4] = {};
    // Main loop: one 32-wide K slice (two NVFP4 scale blocks) per iteration.
    for (int k0 = 0; k0 < k; k0 += kTileK) {
        // Decode A: nibble of element gk -> E2M1 value, times its E4M3 block scale
        // (swizzled). The global scales are applied once, in the epilogue.
        for (int i = tid; i < kTileM * kTileK; i += kThreads) {
            const int r = i / kTileK, kk = i % kTileK;
            const int gr = row0 + r, gk = k0 + kk;
            a_tile[kk][r] = gr < m ? e2m1ToFloat((qa[static_cast<size_t>(gr) * (k / 2) + (gk) / 2] >> (((gk) & 1) * 4)) & 0xFu) * e4m3ToFloat(sa[swizzledScaleIndex(gr, gk / 16, scale_cols)]) : 0.0f;
        }
        // Decode B the same way (B is stored n x k: an "NT" product).
        for (int i = tid; i < kTileN * kTileK; i += kThreads) {
            const int cc = i / kTileK, kk = i % kTileK;
            const int gc = col0 + cc, gk = k0 + kk;
            b_tile[kk][cc] = gc < n ? e2m1ToFloat((qb[static_cast<size_t>(gc) * (k / 2) + (gk) / 2] >> (((gk) & 1) * 4)) & 0xFu) * e4m3ToFloat(sb[swizzledScaleIndex(gc, gk / 16, scale_cols)]) : 0.0f;
        }
        // Tiles complete before anyone reads them.
        __syncthreads();
        // Register-blocked inner product: 4 + 4 shared loads feed 16 FMAs.
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
        // Everyone is done with the tiles before the next slice overwrites them.
        __syncthreads();
    }
    // Epilogue: multiply by 1 / (g_a g_b) and convert the fp32 accumulator to fp16.
    for (int i = 0; i < 4; ++i) {
        const int r = row0 + ty + 16 * i;
        if (r >= m) continue;
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + tx + 16 * j;
            if (col < n) c[static_cast<size_t>(r) * n + col] = __float2half(acc[i][j] * global_scale);
        }
    }
}

// q_a, scale_a, q_b, scale_b, c are device pointers
extern "C" void solution(const uint8_t* q_a, const uint8_t* scale_a, const float sf_g_a, const uint8_t* q_b, const uint8_t* scale_b, const float sf_g_b, __half* c, size_t m, size_t n, size_t k) {
    // One block per 64 x 64 output tile; both global encode factors are folded into one scale.
    const dim3 grid(static_cast<unsigned>((n + kTileN - 1) / kTileN), static_cast<unsigned>((m + kTileM - 1) / kTileM));
    blockScaledGemm<<<grid, kThreads>>>(q_a, scale_a, q_b, scale_b, c, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), (1.0f / sf_g_a) * (1.0f / sf_g_b));
}

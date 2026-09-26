// NVFP4 GEMV (Tensara)
// https://tensara.org/problems/nvfp4-gemv
//
// y (fp16) = dequant(A) dequant(x) with NVFP4 operands (E2M1 values, E4M3 scale
// per 16 elements in the swizzled 128x4 layout, fp32 global scales). GEMV is
// bandwidth-bound, and NVFP4 is ~0.56 bytes per weight, so streaming A packed
// is the whole game: one warp per row, each lane decodes 16-element blocks
// (8 bytes + 1 scale) and accumulates in fp32; the global scales are applied
// once at the end.
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

__global__ void gemv(const uint8_t* __restrict__ qa, const uint8_t* __restrict__ sa, const uint8_t* __restrict__ qx,
                     const uint8_t* __restrict__ sx, __half* __restrict__ y, size_t m, size_t k, float global) {
    const unsigned int lane = threadIdx.x % 32;
    const size_t row = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (row >= m) return;
    const size_t blocks = k / 16;
    float acc = 0.0f;
    for (size_t b = lane; b < blocks; b += 32) {
        const float s = e4m3ToFloat(sa[swizzledScaleIndex(row, b, blocks)]) * e4m3ToFloat(sx[swizzledScaleIndex(0, b, blocks)]);
        const uint8_t* ar = qa + row * (k / 2) + b * 8;
        const uint8_t* xr = qx + b * 8;
        float part = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned int av = ar[j], xv = xr[j];
            part = fmaf(e2m1ToFloat(av & 0xFu), e2m1ToFloat(xv & 0xFu), part);
            part = fmaf(e2m1ToFloat(av >> 4), e2m1ToFloat(xv >> 4), part);
        }
        acc = fmaf(part, s, acc);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = __float2half(acc * global);
}

// q_a, scale_a, q_x, scale_x, y are device pointers
extern "C" void solution(const uint8_t* q_a, const uint8_t* scale_a, const float sf_g_a, const uint8_t* q_x, const uint8_t* scale_x,
                         const float sf_g_x, __half* y, size_t m, size_t k) {
    const size_t threads = m * 32;
    gemv<<<static_cast<unsigned>((threads + 255) / 256), 256>>>(q_a, scale_a, q_x, scale_x, y, m, k, (1.0f / sf_g_a) * (1.0f / sf_g_x));
}

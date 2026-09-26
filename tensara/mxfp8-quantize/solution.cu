// MXFP8 Quantize (Tensara)
// https://tensara.org/problems/mxfp8-quantize
//
// OCP MX quantization with 32-element blocks, matching TorchAO to_mx (FLOOR):
//   scale exponent = floor(log2(amax)) - 8 (the largest power of two of the
//   element format), read straight from the float bits, clamped to [-127, 128];
//   stored as E8M0 = exponent + 127; elements x / 2^exponent -> E4M3 (clamped to +-448, round to nearest even).
// One warp per 32-element block: the block amax is a warp max reduction and
// every lane encodes its own element.
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

__global__ void quantizeMx(const float* __restrict__ a, uint8_t* __restrict__ q, uint8_t* __restrict__ scale, size_t m, size_t k) {
    // One warp per 32-element block: lane l owns element l of the block.
    const unsigned int lane = threadIdx.x % 32;
    const size_t blocks_per_row = k / 32;
    const size_t num_blocks = m * blocks_per_row;
    // Grid-stride loop over blocks (warp-uniform, so the shuffles below are safe).
    for (size_t blk = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32; blk < num_blocks;
         blk += static_cast<size_t>(gridDim.x) * blockDim.x / 32) {
        const size_t row = blk / blocks_per_row;
        const size_t col = (blk % blocks_per_row) * 32 + lane;
        // Block absolute maximum via a butterfly max over the 32 lanes.
        const float x = a[row * k + col];
        float amax = fabsf(x);
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        // Shared exponent = floor(log2(amax)) - 8 (E4M3's largest power of two is 2^8),
        // read from the float's exponent bits and clamped to the E8M0 range.
        int exponent = static_cast<int>((__float_as_uint(amax) >> 23) & 0xFFu) - 127 - 8;
        exponent = exponent < -127 ? -127 : (exponent > 128 ? 128 : exponent);
        const unsigned int biased = static_cast<unsigned int>(exponent + 127);
        // scale as fp32 (biased << 23), clamped to the smallest normal like TorchAO
        const float scale_f = fmaxf(__uint_as_float(biased << 23), 1.17549435e-38f);
        // Scale into E4M3 range, clamp to +-448 and round to nearest even; one byte per element.
        const float v = x / scale_f;
        const unsigned int code = floatToE4M3(fminf(fmaxf(v, -448.0f), 448.0f));
        q[row * k + col] = static_cast<uint8_t>(code);
        // Lane 0 stores the block's E8M0 scale byte (0xFF if the block contains NaN).
        if (lane == 0) scale[blk] = static_cast<uint8_t>(amax != amax ? 255u : biased);
    }
}

// a, q, scale are device pointers
extern "C" void solution(const float* a, uint8_t* q, uint8_t* scale, size_t m, size_t k) {
    // One thread per element (one warp per 32-element block), capped grid.
    size_t blocks = (m * k / 32 * 32 + 255) / 256;
    blocks = blocks > 4096 ? 4096 : (blocks < 1 ? 1 : blocks);
    quantizeMx<<<static_cast<unsigned>(blocks), 256>>>(a, q, scale, m, k);
}

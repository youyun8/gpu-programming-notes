// Edge Detection (Tensara)
// https://tensara.org/problems/edge-detect
//
// Central-difference gradient magnitude sqrt(gx^2 + gy^2), gx = (I[x+1] - I[x-1]) / 2,
// gy = (I[y+1] - I[y-1]) / 2, zero on the border, then scaled so the maximum
// becomes 255.
//   1. magnitude: one thread per pixel; the block max is merged into a global
//      max with atomicMax on the float bits (valid: magnitudes are >= 0, and
//      non-negative IEEE floats order like their integer bit patterns);
//   2. normalize: scale by 255 / max (skipped when the image is flat).
#include <cuda_runtime.h>

constexpr int kThreads = 256;

__device__ unsigned int g_max_bits;

__global__ void resetMax() { g_max_bits = 0u; }

__global__ void magnitude(const float* __restrict__ in, float* __restrict__ out, int h, int w) {
    const size_t total = static_cast<size_t>(h) * w;
    float local_max = 0.0f;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int x = static_cast<int>(idx % w), y = static_cast<int>(idx / w);
        float m = 0.0f;
        if (x > 0 && y > 0 && x < w - 1 && y < h - 1) {
            const float gx = (in[idx + 1] - in[idx - 1]) * 0.5f;
            const float gy = (in[idx + w] - in[idx - w]) * 0.5f;
            m = sqrtf(gx * gx + gy * gy);
        }
        out[idx] = m;
        local_max = fmaxf(local_max, m);
    }
    for (int o = 16; o > 0; o >>= 1) local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, o));
    if (threadIdx.x % 32 == 0) atomicMax(&g_max_bits, __float_as_uint(local_max));
}

__global__ void normalize(float* out, size_t total) {
    const float mx = __uint_as_float(g_max_bits);
    if (!(mx > 0.0f)) return;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(gridDim.x) * blockDim.x)
        out[idx] = out[idx] / mx * 255.0f;
}

// input_image, output_image are device pointers
extern "C" void solution(const float* input_image, float* output_image, size_t height, size_t width) {
    const size_t total = height * width;
    size_t blocks = (total + kThreads - 1) / kThreads;
    blocks = blocks > 4096 ? 4096 : blocks;
    resetMax<<<1, 1>>>();
    magnitude<<<static_cast<unsigned>(blocks), kThreads>>>(input_image, output_image, static_cast<int>(height), static_cast<int>(width));
    normalize<<<static_cast<unsigned>(blocks), kThreads>>>(output_image, total);
}

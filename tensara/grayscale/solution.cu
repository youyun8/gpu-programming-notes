// Grayscale Conversion (Tensara)
// https://tensara.org/problems/grayscale
//
// gray = 0.299 R + 0.587 G + 0.114 B for an interleaved H x W x C image.
// One thread per pixel; a warp's strided channel loads cover one contiguous
// range, so each cache line is fetched once.
#include <cuda_runtime.h>

constexpr int kThreads = 256;

__global__ void grayscaleKernel(const float* __restrict__ rgb, float* __restrict__ gray, size_t pixels, size_t channels) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < pixels; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float* p = rgb + i * channels;
        gray[i] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    }
}

// rgb_image, grayscale_output are device pointers
extern "C" void solution(const float* rgb_image, float* grayscale_output, size_t height, size_t width, size_t channels) {
    const size_t pixels = height * width;
    size_t blocks = (pixels + kThreads - 1) / kThreads;
    blocks = blocks > 65535 ? 65535 : blocks;
    grayscaleKernel<<<static_cast<unsigned>(blocks), kThreads>>>(rgb_image, grayscale_output, pixels, channels);
}

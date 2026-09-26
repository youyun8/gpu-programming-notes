// RGB to Grayscale (LeetGPU)
// https://leetgpu.com/challenges/rgb-to-grayscale
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;

// One thread per pixel. The 3 strided loads of neighbouring threads fall into
// the same cache lines, so global traffic is still ~1x the input size.
__global__ void grayscaleKernel(const float* input, float* output, int num_pixels) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        const float* p = input + 3 * static_cast<size_t>(idx);
        output[idx] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    const int num_pixels = width * height;
    const int num_blocks = (num_pixels + kBlockSize - 1) / kBlockSize;
    grayscaleKernel<<<num_blocks, kBlockSize>>>(input, output, num_pixels);
    cudaDeviceSynchronize();
}

// Color Inversion (LeetGPU)
// https://leetgpu.com/challenges/color-inversion
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;

// One thread per RGBA pixel, loaded as a single 32-bit uchar4.
__global__ void invertColors(uchar4* pixels, int num_pixels) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        uchar4 p = pixels[idx];
        p.x = 255 - p.x;
        p.y = 255 - p.y;
        p.z = 255 - p.z;
        pixels[idx] = p;
    }
}

// image is a device pointer
extern "C" void solve(unsigned char* image, int width, int height) {
    const int num_pixels = width * height;
    const int num_blocks = (num_pixels + kBlockSize - 1) / kBlockSize;
    invertColors<<<num_blocks, kBlockSize>>>(reinterpret_cast<uchar4*>(image), num_pixels);
    cudaDeviceSynchronize();
}

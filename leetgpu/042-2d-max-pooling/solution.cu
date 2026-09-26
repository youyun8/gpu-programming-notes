// 2D Max Pooling (LeetGPU)
// https://leetgpu.com/challenges/2d-max-pooling
//
// One thread per output element (n, c, oh, ow); padded positions are skipped,
// which is equivalent to padding with -inf as PyTorch does.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kBlockSize = 256;

__global__ void maxPool2d(const float* input, float* output, int nc, int h, int w, int oh, int ow, int k, int stride,
                          int pad) {
    const size_t total = static_cast<size_t>(nc) * oh * ow;
    for (size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; idx < total;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int x = static_cast<int>(idx % ow);
        const int y = static_cast<int>((idx / ow) % oh);
        const size_t plane = idx / (static_cast<size_t>(ow) * oh);
        const float* in = input + plane * h * w;
        const int y0 = y * stride - pad;
        const int x0 = x * stride - pad;
        float best = -FLT_MAX;
        for (int dy = 0; dy < k; ++dy) {
            const int iy = y0 + dy;
            if (iy < 0 || iy >= h) continue;
            for (int dx = 0; dx < k; ++dx) {
                const int ix = x0 + dx;
                if (ix >= 0 && ix < w) best = fmaxf(best, in[static_cast<size_t>(iy) * w + ix]);
            }
        }
        output[idx] = best;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int C, int H, int W, int kernel_size, int stride,
                      int padding) {
    const int oh = (H + 2 * padding - kernel_size) / stride + 1;
    const int ow = (W + 2 * padding - kernel_size) / stride + 1;
    const size_t total = static_cast<size_t>(N) * C * oh * ow;
    int blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    blocks = blocks > 65535 ? 65535 : (blocks < 1 ? 1 : blocks);
    maxPool2d<<<blocks, kBlockSize>>>(input, output, N * C, H, W, oh, ow, kernel_size, stride, padding);
    cudaDeviceSynchronize();
}

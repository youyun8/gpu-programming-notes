// Value Clipping (LeetGPU)
// https://leetgpu.com/challenges/value-clipping
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;

__global__ void clipKernel(const float* input, float* output, float lo, float hi, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = fminf(fmaxf(input[idx], lo), hi);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    const int num_blocks = (N + kBlockSize - 1) / kBlockSize;
    clipKernel<<<num_blocks, kBlockSize>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}

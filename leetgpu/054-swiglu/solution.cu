// SwiGLU (LeetGPU)
// https://leetgpu.com/challenges/swish-gated-linear-unit
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;

// output[i] = silu(x[i]) * x[i + N/2]
__global__ void swigluKernel(const float* input, float* output, int half_n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < half_n) {
        const float x1 = input[idx];
        const float x2 = input[idx + half_n];
        output[idx] = x1 / (1.0f + expf(-x1)) * x2;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    const int half_n = N / 2;
    const int num_blocks = (half_n + kBlockSize - 1) / kBlockSize;
    if (num_blocks > 0) swigluKernel<<<num_blocks, kBlockSize>>>(input, output, half_n);
    cudaDeviceSynchronize();
}

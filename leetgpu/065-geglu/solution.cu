// GEGLU (LeetGPU)
// https://leetgpu.com/challenges/geglu
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr float kInvSqrt2 = 0.70710678118654752f;

// output[i] = x1 * gelu(x2) with the exact (erf) GELU.
__global__ void gegluKernel(const float* input, float* output, int half_n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < half_n) {
        const float x1 = input[idx];
        const float x2 = input[idx + half_n];
        output[idx] = x1 * (0.5f * x2 * (1.0f + erff(x2 * kInvSqrt2)));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    const int half_n = N / 2;
    const int num_blocks = (half_n + kBlockSize - 1) / kBlockSize;
    if (num_blocks > 0) gegluKernel<<<num_blocks, kBlockSize>>>(input, output, half_n);
    cudaDeviceSynchronize();
}

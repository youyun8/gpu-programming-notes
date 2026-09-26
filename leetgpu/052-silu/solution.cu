// SiLU (LeetGPU)
// https://leetgpu.com/challenges/sigmoid-linear-unit
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;

// x * sigmoid(x) = x / (1 + e^-x). For x -> -inf, e^-x overflows to inf and
// the quotient correctly becomes -0.
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__global__ void siluKernel(const float* input, float* output, int n) {
    // One thread per element.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = silu(input[idx]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    const int num_blocks = (N + kBlockSize - 1) / kBlockSize;
    siluKernel<<<num_blocks, kBlockSize>>>(input, output, N);
    cudaDeviceSynchronize();
}

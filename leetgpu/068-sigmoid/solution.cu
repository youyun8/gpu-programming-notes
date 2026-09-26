// Sigmoid (LeetGPU)
// https://leetgpu.com/challenges/sigmoid-activation
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void sigmoidKernel(const float* x, float* y, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_vec4 = n / 4;
    // Bulk: one float4 per thread; the first n % 4 threads also handle one tail element.
    if (idx < num_vec4) {
        const float4 v = reinterpret_cast<const float4*>(x)[idx];
        reinterpret_cast<float4*>(y)[idx] = make_float4(sigmoid(v.x), sigmoid(v.y), sigmoid(v.z), sigmoid(v.w));
    }
    if (idx < n % 4) y[num_vec4 * 4 + idx] = sigmoid(x[num_vec4 * 4 + idx]);
}

// X, Y are device pointers
extern "C" void solve(const float* X, float* Y, int N) {
    // Enough threads for every float4, and at least one block for the tail.
    const int num_blocks = (N / 4 + kBlockSize) / kBlockSize;
    sigmoidKernel<<<num_blocks, kBlockSize>>>(X, Y, N);
    cudaDeviceSynchronize();
}

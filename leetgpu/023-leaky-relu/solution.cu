// Leaky ReLU (LeetGPU)
// https://leetgpu.com/challenges/leaky-relu
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr float kAlpha = 0.01f;

__device__ __forceinline__ float leakyRelu(float x) { return x > 0.0f ? x : kAlpha * x; }

__global__ void leakyReluKernel(const float* input, float* output, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_vec4 = n / 4;
    // Bulk: one float4 per thread.
    if (idx < num_vec4) {
        const float4 v = reinterpret_cast<const float4*>(input)[idx];
        reinterpret_cast<float4*>(output)[idx] =
            make_float4(leakyRelu(v.x), leakyRelu(v.y), leakyRelu(v.z), leakyRelu(v.w));
    }
    // Tail: the first n % 4 threads also handle one leftover element.
    if (idx < n % 4) output[num_vec4 * 4 + idx] = leakyRelu(input[num_vec4 * 4 + idx]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    // Enough threads for every float4, and at least one block for the tail.
    const int num_blocks = (N / 4 + kBlockSize) / kBlockSize;
    leakyReluKernel<<<num_blocks, kBlockSize>>>(input, output, N);
    cudaDeviceSynchronize();
}

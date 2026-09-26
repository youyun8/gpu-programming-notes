// 1D Convolution (LeetGPU)
// https://leetgpu.com/challenges/1d-convolution
//
// output[i] = sum_j input[i + j] * kernel[j]  ("valid" cross-correlation)
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kOutputsPerThread = 4;
constexpr int kOutputsPerBlock = kBlockSize * kOutputsPerThread;

// Each block stages its input window and the whole kernel in shared memory.
// Thread t computes outputs t, t + 256, t + 512, t + 768 of the block so that
// neighbouring threads read neighbouring shared-memory words.
__global__ void conv1d(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    extern __shared__ float smem[];
    float* s_kernel = smem;
    float* s_input = smem + kernel_size;

    const int out_size = input_size - kernel_size + 1;
    const int base = blockIdx.x * kOutputsPerBlock;
    const int window = kOutputsPerBlock + kernel_size - 1;

    for (int i = threadIdx.x; i < kernel_size; i += kBlockSize) s_kernel[i] = kernel[i];
    for (int i = threadIdx.x; i < window; i += kBlockSize) {
        const int g = base + i;
        s_input[i] = g < input_size ? input[g] : 0.0f;
    }
    __syncthreads();

    float acc[kOutputsPerThread] = {};
    for (int j = 0; j < kernel_size; ++j) {
        const float w = s_kernel[j];
#pragma unroll
        for (int r = 0; r < kOutputsPerThread; ++r) acc[r] = fmaf(s_input[threadIdx.x + r * kBlockSize + j], w, acc[r]);
    }
#pragma unroll
    for (int r = 0; r < kOutputsPerThread; ++r) {
        const int o = base + threadIdx.x + r * kBlockSize;
        if (o < out_size) output[o] = acc[r];
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    const int out_size = input_size - kernel_size + 1;
    const int num_blocks = (out_size + kOutputsPerBlock - 1) / kOutputsPerBlock;
    const size_t smem = (2 * static_cast<size_t>(kernel_size) - 1 + kOutputsPerBlock) * sizeof(float);
    conv1d<<<num_blocks, kBlockSize, smem>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}

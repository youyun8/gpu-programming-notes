// Matrix Copy (LeetGPU)
// https://leetgpu.com/challenges/matrix-copy
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;

// Pure bandwidth test: copy N*N floats with 16-byte loads/stores.
__global__ void copyKernel(const float* src, float* dst, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_vec4 = total / 4;
    if (idx < num_vec4) reinterpret_cast<float4*>(dst)[idx] = reinterpret_cast<const float4*>(src)[idx];
    if (idx < total % 4) dst[num_vec4 * 4 + idx] = src[num_vec4 * 4 + idx];
}

// A, B are device pointers
extern "C" void solve(const float* A, float* B, int N) {
    const int total = N * N;
    const int num_blocks = (total / 4 + kBlockSize) / kBlockSize;
    copyKernel<<<num_blocks, kBlockSize>>>(A, B, total);
    cudaDeviceSynchronize();
}

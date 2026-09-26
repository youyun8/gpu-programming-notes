// Vector Addition (LeetGPU)
// https://leetgpu.com/challenges/vector-addition
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// a, b, c are device pointers
extern "C" void solve(const float* a, const float* b, float* c, int n) {
    const int num_blocks = (n + kBlockSize - 1) / kBlockSize;
    vectorAdd<<<num_blocks, kBlockSize>>>(a, b, c, n);
    cudaDeviceSynchronize();
}

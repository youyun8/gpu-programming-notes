// Vector Addition (LeetGPU)
// https://leetgpu.com/challenges/vector-addition
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    // One thread per element; the last block may be partial, so guard the index.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    // ceil(N / 256) blocks cover every element.
    const int num_blocks = (N + kBlockSize - 1) / kBlockSize;
    vectorAdd<<<num_blocks, kBlockSize>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

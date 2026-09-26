// Interleave (LeetGPU)
// https://leetgpu.com/challenges/interleave
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;

// Each thread reads one float from A and one from B and writes them as a single
// 8-byte float2, so both the loads and the store are fully coalesced.
__global__ void interleaveKernel(const float* a, const float* b, float2* output, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = make_float2(a[idx], b[idx]);
}

// A, B, output are device pointers
extern "C" void solve(const float* A, const float* B, float* output, int N) {
    const int num_blocks = (N + kBlockSize - 1) / kBlockSize;
    interleaveKernel<<<num_blocks, kBlockSize>>>(A, B, reinterpret_cast<float2*>(output), N);
    cudaDeviceSynchronize();
}

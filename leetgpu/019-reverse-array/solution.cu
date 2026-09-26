// Reverse Array (LeetGPU)
// https://leetgpu.com/challenges/reverse-array
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;

// Thread i swaps element i with its mirror; only the first half does work so
// each pair is touched exactly once (no race on the in-place update).
__global__ void reverseInPlace(float* data, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n / 2) {
        const int j = n - 1 - i;
        const float tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

// input is a device pointer
extern "C" void solve(float* input, int N) {
    // One thread per pair; nothing to do for N < 2.
    const int half_n = N / 2;
    const int num_blocks = (half_n + kBlockSize - 1) / kBlockSize;
    if (num_blocks > 0) reverseInPlace<<<num_blocks, kBlockSize>>>(input, N);
    cudaDeviceSynchronize();
}

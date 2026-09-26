// Rainbow Table (LeetGPU)
// https://leetgpu.com/challenges/rainbow-table
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr unsigned int kFnvPrime = 16777619u;
constexpr unsigned int kFnvOffsetBasis = 2166136261u;

// 32-bit FNV-1a over the 4 little-endian bytes of x. Unsigned overflow gives
// exactly the "& 0xFFFFFFFF" of the reference.
__device__ __forceinline__ unsigned int fnv1a(unsigned int x) {
    unsigned int hash = kFnvOffsetBasis;
#pragma unroll
    for (int byte_pos = 0; byte_pos < 4; ++byte_pos) {
        hash = (hash ^ ((x >> (8 * byte_pos)) & 0xFFu)) * kFnvPrime;
    }
    return hash;
}

// Compute-bound: every element is independent, the R rounds stay in registers.
__global__ void rainbowKernel(const int* input, unsigned int* output, int n, int rounds) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Apply the hash R times to the element's bit pattern.
        unsigned int h = static_cast<unsigned int>(input[idx]);
        for (int r = 0; r < rounds; ++r) h = fnv1a(h);
        output[idx] = h;
    }
}

// input, output are device pointers
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    const int num_blocks = (N + kBlockSize - 1) / kBlockSize;
    rainbowKernel<<<num_blocks, kBlockSize>>>(input, output, N, R);
    cudaDeviceSynchronize();
}

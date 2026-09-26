// Subarray Sum (LeetGPU)
// https://leetgpu.com/challenges/subarray-sum
//
// Sum of input[S..E]: an integer grid-stride reduction over the range with a
// warp-level __reduce_add_sync and one atomicAdd per warp. Integer addition is
// associative, so the result is exact and order-independent.
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 2048;

__global__ void rangeSum(const int* input, int* output, int s, int e) {
    const int len = e - s + 1;
    int local = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x) local += input[s + i];
    local = __reduce_add_sync(0xffffffffu, local);
    if (threadIdx.x % 32 == 0 && local) atomicAdd(output, local);
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    cudaMemset(output, 0, sizeof(int));
    int blocks = (E - S + 1 + kBlockSize - 1) / kBlockSize;
    blocks = blocks > kMaxBlocks ? kMaxBlocks : blocks;
    rangeSum<<<blocks, kBlockSize>>>(input, output, S, E);
    cudaDeviceSynchronize();
}

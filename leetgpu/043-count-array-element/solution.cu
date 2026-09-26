// Count Array Element (LeetGPU)
// https://leetgpu.com/challenges/count-array-element
//
// Counting reduction: int4 loads, per-thread counts, warp-level
// __reduce_add_sync (sm_80+ integer warp reduction), one atomicAdd per warp.
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 2048;

__global__ void countEqual(const int* input, int* output, int n, int k) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    int count = 0;
    for (int i = tid; i < n / 4; i += stride) {
        const int4 v = reinterpret_cast<const int4*>(input)[i];
        count += (v.x == k) + (v.y == k) + (v.z == k) + (v.w == k);
    }
    for (int i = (n / 4) * 4 + tid; i < n; i += stride) count += input[i] == k;
    count = __reduce_add_sync(0xffffffffu, count);
    if (threadIdx.x % 32 == 0 && count) atomicAdd(output, count);
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int K) {
    cudaMemset(output, 0, sizeof(int));
    int blocks = (N / 4 + kBlockSize - 1) / kBlockSize;
    blocks = blocks < 1 ? 1 : (blocks > kMaxBlocks ? kMaxBlocks : blocks);
    countEqual<<<blocks, kBlockSize>>>(input, output, N, K);
    cudaDeviceSynchronize();
}

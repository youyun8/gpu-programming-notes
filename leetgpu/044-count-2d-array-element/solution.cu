// Count 2D Array Element (LeetGPU)
// https://leetgpu.com/challenges/count-2d-array-element
//
// The array is contiguous, so the dimensions only define the element count:
// a flat counting reduction with int4 loads, a warp-level __reduce_add_sync
// and one atomicAdd per warp.
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 2048;

__global__ void countEqual(const int* input, int* output, long long n, int key) {
    const long long tid = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    // Grid-stride count: compare four ints per 16-byte int4 load, then a scalar tail.
    int count = 0;
    for (long long i = tid; i < n / 4; i += stride) {
        const int4 v = reinterpret_cast<const int4*>(input)[i];
        count += (v.x == key) + (v.y == key) + (v.z == key) + (v.w == key);
    }
    for (long long i = (n / 4) * 4 + tid; i < n; i += stride) count += input[i] == key;
    // Warp-wide integer sum in one instruction (sm_80+); one atomic per warp with a non-zero count.
    count = __reduce_add_sync(0xffffffffu, count);
    if (threadIdx.x % 32 == 0 && count) atomicAdd(output, count);
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    // N * M elements, in 64-bit arithmetic.
    const long long total = static_cast<long long>(N) * M;
    // The result is accumulated with atomics: clear it first.
    cudaMemset(output, 0, sizeof(int));
    long long blocks = (total / 4 + kBlockSize - 1) / kBlockSize;
    blocks = blocks < 1 ? 1 : (blocks > kMaxBlocks ? kMaxBlocks : blocks);
    countEqual<<<static_cast<int>(blocks), kBlockSize>>>(input, output, total, K);
    cudaDeviceSynchronize();
}

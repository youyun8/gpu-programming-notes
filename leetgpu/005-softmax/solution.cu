// Softmax (LeetGPU)
// https://leetgpu.com/challenges/softmax
//
// 1. Each block computes an "online softmax" pair (max m, sum of e^(x-m)) for
//    its grid-stride slice.
// 2. One block merges the pairs: (m1,s1) + (m2,s2) = (m, s1 e^(m1-m) + s2 e^(m2-m)).
// 3. An elementwise kernel writes e^(x - M) / S.
// Two reads + one write of the input instead of the naive three passes.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 512;

__device__ float g_block_max[kMaxBlocks];
__device__ float g_block_sum[kMaxBlocks];
__device__ float g_max;
__device__ float g_sum;

struct MaxSum {
    float m;
    float s;
};

__device__ __forceinline__ MaxSum combine(MaxSum a, MaxSum b) {
    const float m = fmaxf(a.m, b.m);
    if (m == -FLT_MAX) return {m, 0.0f};  // both empty
    return {m, a.s * expf(a.m - m) + b.s * expf(b.m - m)};
}

__device__ MaxSum blockCombine(MaxSum v) {
    __shared__ MaxSum warp_vals[32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset >>= 1) {
        MaxSum other{__shfl_down_sync(0xffffffffu, v.m, offset), __shfl_down_sync(0xffffffffu, v.s, offset)};
        v = combine(v, other);
    }
    if (lane == 0) warp_vals[warp] = v;
    __syncthreads();
    v = threadIdx.x < blockDim.x / 32 ? warp_vals[lane] : MaxSum{-FLT_MAX, 0.0f};
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            MaxSum other{__shfl_down_sync(0xffffffffu, v.m, offset), __shfl_down_sync(0xffffffffu, v.s, offset)};
            v = combine(v, other);
        }
    }
    return v;
}

__global__ void blockMaxSum(const float* input, int n) {
    MaxSum v{-FLT_MAX, 0.0f};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        v = combine(v, MaxSum{input[i], 1.0f});
    }
    v = blockCombine(v);
    if (threadIdx.x == 0) {
        g_block_max[blockIdx.x] = v.m;
        g_block_sum[blockIdx.x] = v.s;
    }
}

__global__ void globalMaxSum(int num_blocks) {
    MaxSum v{-FLT_MAX, 0.0f};
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) v = combine(v, MaxSum{g_block_max[i], g_block_sum[i]});
    v = blockCombine(v);
    if (threadIdx.x == 0) {
        g_max = v.m;
        g_sum = v.s;
    }
}

__global__ void normalize(const float* input, float* output, int n) {
    const float m = g_max;
    const float inv_sum = 1.0f / g_sum;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        output[i] = expf(input[i] - m) * inv_sum;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int num_blocks = (N + kBlockSize - 1) / kBlockSize;
    num_blocks = num_blocks > kMaxBlocks ? kMaxBlocks : num_blocks;
    blockMaxSum<<<num_blocks, kBlockSize>>>(input, N);
    globalMaxSum<<<1, kBlockSize>>>(num_blocks);
    normalize<<<num_blocks, kBlockSize>>>(input, output, N);
    cudaDeviceSynchronize();
}

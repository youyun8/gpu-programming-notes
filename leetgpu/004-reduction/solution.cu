// Reduction (LeetGPU)
// https://leetgpu.com/challenges/reduction
//
// Two-pass, deterministic sum:
//   1. each block reduces a grid-stride slice (float4 loads, fp32 per thread)
//      and writes one fp64 partial;
//   2. a single block adds the partials in fp64 and writes the float result.
// fp64 for the cross-thread part keeps the error far below the 1e-5 tolerance
// (the reference sums in double) at no bandwidth cost.
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 1024;

__device__ double g_partials[kMaxBlocks];

__device__ __forceinline__ double warpReduceSum(double v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// Result is valid in thread 0. blockDim.x must be a multiple of 32.
__device__ double blockReduceSum(double v) {
    __shared__ double warp_sums[32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    v = warpReduceSum(v);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    v = threadIdx.x < blockDim.x / 32 ? warp_sums[lane] : 0.0;
    if (warp == 0) v = warpReduceSum(v);
    return v;
}

__global__ void partialSums(const float* input, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int num_vec4 = n / 4;
    float local = 0.0f;
    for (int i = tid; i < num_vec4; i += stride) {
        const float4 v = reinterpret_cast<const float4*>(input)[i];
        local += (v.x + v.y) + (v.z + v.w);
    }
    for (int i = num_vec4 * 4 + tid; i < n; i += stride) local += input[i];
    const double block_sum = blockReduceSum(local);
    if (threadIdx.x == 0) g_partials[blockIdx.x] = block_sum;
}

__global__ void finalSum(float* output, int num_partials) {
    double v = 0.0;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) v += g_partials[i];
    v = blockReduceSum(v);
    if (threadIdx.x == 0) output[0] = static_cast<float>(v);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int num_blocks = (N / 4 + kBlockSize - 1) / kBlockSize;
    num_blocks = num_blocks < 1 ? 1 : (num_blocks > kMaxBlocks ? kMaxBlocks : num_blocks);
    partialSums<<<num_blocks, kBlockSize>>>(input, N);
    finalSum<<<1, kBlockSize>>>(output, num_blocks);
    cudaDeviceSynchronize();
}

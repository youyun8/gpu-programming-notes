// Dot Product (LeetGPU)
// https://leetgpu.com/challenges/dot-product
//
// Same two-pass scheme as the reduction problem: fp32 FMAs per thread,
// fp64 block partials, one final block writes the float result.
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 1024;

// One fp64 partial per block of the first pass.
__device__ double g_partials[kMaxBlocks];

__device__ __forceinline__ double warpReduceSum(double v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__device__ double blockReduceSum(double v) {
    __shared__ double warp_sums[32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    // Reduce inside each warp; lane 0 publishes the warp sum.
    v = warpReduceSum(v);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    // Warp 0 reduces the warp sums.
    v = threadIdx.x < blockDim.x / 32 ? warp_sums[lane] : 0.0;
    if (warp == 0) v = warpReduceSum(v);
    return v;
}

__global__ void partialDots(const float* a, const float* b, int n) {
    // Pass 1: grid-stride FMAs over float4 pairs plus a scalar tail, then a block reduction in double.
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int num_vec4 = n / 4;
    float local = 0.0f;
    for (int i = tid; i < num_vec4; i += stride) {
        const float4 x = reinterpret_cast<const float4*>(a)[i];
        const float4 y = reinterpret_cast<const float4*>(b)[i];
        local = fmaf(x.x, y.x, local);
        local = fmaf(x.y, y.y, local);
        local = fmaf(x.z, y.z, local);
        local = fmaf(x.w, y.w, local);
    }
    for (int i = num_vec4 * 4 + tid; i < n; i += stride) local = fmaf(a[i], b[i], local);
    const double block_sum = blockReduceSum(local);
    if (threadIdx.x == 0) g_partials[blockIdx.x] = block_sum;
}

__global__ void finalSum(float* result, int num_partials) {
    // Pass 2 (one block): add the partials in double.
    double v = 0.0;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) v += g_partials[i];
    v = blockReduceSum(v);
    if (threadIdx.x == 0) result[0] = static_cast<float>(v);
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    // Two launches, no atomics: deterministic.
    int num_blocks = (N / 4 + kBlockSize - 1) / kBlockSize;
    num_blocks = num_blocks < 1 ? 1 : (num_blocks > kMaxBlocks ? kMaxBlocks : num_blocks);
    partialDots<<<num_blocks, kBlockSize>>>(A, B, N);
    finalSum<<<1, kBlockSize>>>(result, num_blocks);
    cudaDeviceSynchronize();
}

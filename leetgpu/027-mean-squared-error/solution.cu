// Mean Squared Error (LeetGPU)
// https://leetgpu.com/challenges/mean-squared-error
//
// Two-pass reduction: fp32 per thread over a grid-stride slice (float4 loads),
// fp64 block partials, a final block divides by N.
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 1024;

// One fp64 partial per block of the first pass.
__device__ double g_partials[kMaxBlocks];

__device__ double blockReduceSum(double v) {
    __shared__ double warp_sums[32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    // Block-wide sum: warp shuffle trees, then warp 0 reduces the warp sums (result in thread 0).
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    v = threadIdx.x < blockDim.x / 32 ? warp_sums[lane] : 0.0;
    if (warp == 0)
        for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__device__ __forceinline__ float sq(float x) { return x * x; }

__global__ void partialSquares(const float* p, const float* t, int n) {
    // Pass 1: grid-stride sum of squared differences (float4 plus a scalar tail), block-reduced in double.
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int num_vec4 = n / 4;
    float local = 0.0f;
    for (int i = tid; i < num_vec4; i += stride) {
        const float4 a = reinterpret_cast<const float4*>(p)[i];
        const float4 b = reinterpret_cast<const float4*>(t)[i];
        local += (sq(a.x - b.x) + sq(a.y - b.y)) + (sq(a.z - b.z) + sq(a.w - b.w));
    }
    for (int i = num_vec4 * 4 + tid; i < n; i += stride) local += sq(p[i] - t[i]);
    const double s = blockReduceSum(local);
    if (threadIdx.x == 0) g_partials[blockIdx.x] = s;
}

__global__ void finalMean(float* mse, int num_partials, int n) {
    // Pass 2 (one block): add the partials and divide by N.
    double v = 0.0;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) v += g_partials[i];
    v = blockReduceSum(v);
    if (threadIdx.x == 0) mse[0] = static_cast<float>(v / n);
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    int num_blocks = (N / 4 + kBlockSize - 1) / kBlockSize;
    num_blocks = num_blocks < 1 ? 1 : (num_blocks > kMaxBlocks ? kMaxBlocks : num_blocks);
    partialSquares<<<num_blocks, kBlockSize>>>(predictions, targets, N);
    finalMean<<<1, kBlockSize>>>(mse, num_blocks, N);
    cudaDeviceSynchronize();
}

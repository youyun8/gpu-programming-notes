// Monte Carlo Integration (LeetGPU)
// https://leetgpu.com/challenges/monte-carlo-integration
//
// integral ~= (b - a) * mean(y): a two-pass sum reduction (float4 loads,
// fp64 block partials) followed by one scaling step.
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 1024;

__device__ double g_partials[kMaxBlocks];

__device__ double blockReduceSum(double v) {
    __shared__ double warp_sums[32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    v = threadIdx.x < blockDim.x / 32 ? warp_sums[lane] : 0.0;
    if (warp == 0)
        for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__global__ void partialSums(const float* y, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    float local = 0.0f;
    for (int i = tid; i < n / 4; i += stride) {
        const float4 v = reinterpret_cast<const float4*>(y)[i];
        local += (v.x + v.y) + (v.z + v.w);
    }
    for (int i = (n / 4) * 4 + tid; i < n; i += stride) local += y[i];
    const double s = blockReduceSum(local);
    if (threadIdx.x == 0) g_partials[blockIdx.x] = s;
}

__global__ void finalize(float* result, int num_partials, float a, float b, int n) {
    double v = 0.0;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) v += g_partials[i];
    v = blockReduceSum(v);
    if (threadIdx.x == 0) result[0] = static_cast<float>((static_cast<double>(b) - a) * (v / n));
}

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    int num_blocks = (n_samples / 4 + kBlockSize - 1) / kBlockSize;
    num_blocks = num_blocks < 1 ? 1 : (num_blocks > kMaxBlocks ? kMaxBlocks : num_blocks);
    partialSums<<<num_blocks, kBlockSize>>>(y_samples, n_samples);
    finalize<<<1, kBlockSize>>>(result, num_blocks, a, b, n_samples);
    cudaDeviceSynchronize();
}

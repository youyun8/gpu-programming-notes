// RMS Normalization (LeetGPU)
// https://leetgpu.com/challenges/rms-normalization
//
// rms = sqrt(mean(x^2) + eps); y = gamma * x / rms + beta over one vector.
// Pass 1: blocks reduce sum(x^2) into fp64 partials; pass 2: one block turns
// them into 1/rms; pass 3: elementwise scale + shift.
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 512;

__device__ double g_partials[kMaxBlocks];
__device__ float g_inv_rms;

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

__global__ void sumSquares(const float* x, int n) {
    float local = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) local = fmaf(x[i], x[i], local);
    const double s = blockReduceSum(local);
    if (threadIdx.x == 0) g_partials[blockIdx.x] = s;
}

__global__ void computeInvRms(int num_partials, int n, float eps) {
    double v = 0.0;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) v += g_partials[i];
    v = blockReduceSum(v);
    if (threadIdx.x == 0) g_inv_rms = static_cast<float>(1.0 / sqrt(v / n + eps));
}

__global__ void scaleShift(const float* x, float* y, int n, float gamma, float beta) {
    const float inv_rms = g_inv_rms;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] = gamma * (x[i] * inv_rms) + beta;
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N, float eps) {
    int blocks = (N + kBlockSize - 1) / kBlockSize;
    blocks = blocks > kMaxBlocks ? kMaxBlocks : blocks;
    sumSquares<<<blocks, kBlockSize>>>(input, N);
    computeInvRms<<<1, kBlockSize>>>(blocks, N, eps);
    scaleShift<<<blocks, kBlockSize>>>(input, output, N, gamma, beta);
    cudaDeviceSynchronize();
}

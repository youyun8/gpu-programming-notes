// PPO Clipped Surrogate Loss (LeetGPU)
// https://leetgpu.com/challenges/ppo-clipped-surrogate-loss
//
// loss = -mean(min(r A, clip(r, 1 - eps, 1 + eps) A)), r = exp(log_pi - log_pi_old).
// Elementwise transform fused into a two-pass reduction (fp64 block partials).
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

__global__ void surrogateSums(const float* adv, const float* log_pi, const float* log_pi_old, int n, float clip_eps) {
    float local = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float r = expf(log_pi[i] - log_pi_old[i]);
        const float rc = fminf(fmaxf(r, 1.0f - clip_eps), 1.0f + clip_eps);
        local += fminf(r * adv[i], rc * adv[i]);
    }
    const double s = blockReduceSum(local);
    if (threadIdx.x == 0) g_partials[blockIdx.x] = s;
}

__global__ void finalize(float* output, int num_partials, int n) {
    double v = 0.0;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) v += g_partials[i];
    v = blockReduceSum(v);
    if (threadIdx.x == 0) output[0] = static_cast<float>(-v / n);
}

// advantages, log_pi, log_pi_old, output are device pointers
extern "C" void solve(const float* advantages, const float* log_pi, const float* log_pi_old, float* output,
                      float clip_eps, int B, int S) {
    const int n = B * S;
    int blocks = (n + kBlockSize - 1) / kBlockSize;
    blocks = blocks > kMaxBlocks ? kMaxBlocks : blocks;
    surrogateSums<<<blocks, kBlockSize>>>(advantages, log_pi, log_pi_old, n, clip_eps);
    finalize<<<1, kBlockSize>>>(output, blocks, n);
    cudaDeviceSynchronize();
}

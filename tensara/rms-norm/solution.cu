// RMS Normalization (Tensara)
// https://tensara.org/problems/rms-norm
//
// y = x / sqrt(mean(x^2) + 1e-5) per row (no weight).
// One block per row: pass 1 reduces the row (block reduction: warp shuffles
// + one shared-memory hop), pass 2 rescales it. The row is re-read from L1/L2.
#include <cuda_runtime.h>

constexpr int kThreads = 256;

__device__ float blockSum(float v) {
    __shared__ float warp_sums[32];
    __shared__ float total;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x % 32 == 0) warp_sums[threadIdx.x / 32] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int w = 0; w < static_cast<int>(blockDim.x / 32); ++w) t += warp_sums[w];
        total = t;
    }
    __syncthreads();
    const float result = total;
    __syncthreads();  // total / warp_sums may be reused by the next call
    return result;
}

__global__ void normalizeRows(const float* __restrict__ x, float* __restrict__ y, size_t d) {
    const float* xr = x + blockIdx.x * d;
    float* yr = y + blockIdx.x * d;
    float s = 0.0f;
    for (size_t j = threadIdx.x; j < d; j += kThreads) {
        const float v = xr[j];
        s += v * v;
    }
    s = blockSum(s);
    const float inv = 1.0f / (sqrtf(s / static_cast<float>(d) + 1e-5f));
    for (size_t j = threadIdx.x; j < d; j += kThreads) yr[j] = xr[j] * inv;
}

// X, Y are device pointers
extern "C" void solution(const float* X, float* Y, size_t B, size_t N) {
    normalizeRows<<<static_cast<unsigned>(B), kThreads>>>(X, Y, N);
}

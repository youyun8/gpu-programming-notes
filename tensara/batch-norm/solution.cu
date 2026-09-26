// Batch Normalization (Tensara)
// https://tensara.org/problems/batch-norm
//
// BatchNorm2d in training mode without affine parameters: per channel f the
// statistics run over (B, D1, D2). For channel f the data is B contiguous
// chunks of D1*D2 floats, so one block per channel walks them with coalesced
// loads; mean and centered variance are reduced in fp64, then normalized.
#include <cuda_runtime.h>

constexpr int kThreads = 1024;
constexpr float kEps = 1e-5f;

__device__ double blockSumD(double v) {
    // Block-wide sum in double; every thread receives the result.
    __shared__ double warp_sums[32];
    __shared__ double total;
    // Butterfly sum inside each warp; lane 0 publishes the warp total.
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x % 32 == 0) warp_sums[threadIdx.x / 32] = v;
    __syncthreads();
    // Thread 0 adds the warp totals and broadcasts the result through shared memory.
    if (threadIdx.x == 0) {
        double t = 0.0;
        for (int w = 0; w < static_cast<int>(blockDim.x / 32); ++w) t += warp_sums[w];
        total = t;
    }
    __syncthreads();
    const double result = total;
    __syncthreads();
    return result;
}

__global__ void batchNorm(const float* x, float* y, size_t b, size_t f, size_t hw) {
    // One block per channel. Its data are B contiguous chunks of hw floats with stride f * hw;
    // the flat index i maps to (batch i / hw, position i % hw), so consecutive threads
    // read consecutive addresses.
    const size_t ch = blockIdx.x;
    const size_t count = b * hw;
    // Pass 1: mean over (B, D1, D2).
    double s = 0.0;
    for (size_t i = threadIdx.x; i < count; i += kThreads) s += x[((i / hw) * f + ch) * hw + i % hw];
    const float mean = static_cast<float>(blockSumD(s) / count);
    // Pass 2: centered sum of squares (no E[x^2] - E[x]^2 cancellation), biased variance.
    double sq = 0.0;
    for (size_t i = threadIdx.x; i < count; i += kThreads) {
        const double t = x[((i / hw) * f + ch) * hw + i % hw] - mean;
        sq += t * t;
    }
    const float rstd = static_cast<float>(1.0 / sqrt(blockSumD(sq) / count + kEps));
    for (size_t i = threadIdx.x; i < count; i += kThreads) {
        // Pass 3: normalize.
        const size_t idx = ((i / hw) * f + ch) * hw + i % hw;
        y[idx] = (x[idx] - mean) * rstd;
    }
}

// X, Y are device pointers
extern "C" void solution(const float* X, float* Y, size_t B, size_t F, size_t D1, size_t D2) {
    // One 1024-thread block per channel.
    batchNorm<<<static_cast<unsigned>(F), kThreads>>>(X, Y, B, F, D1 * D2);
}

// Layer Normalization (Tensara)
// https://tensara.org/problems/layer-norm
//
// LayerNorm over the last three dims (F, D1, D2) of a (B, F, D1, D2) tensor,
// with elementwise gamma / beta of shape (F, D1, D2).
// Each batch element is a contiguous normalization group -> one block per
// group; mean and centered variance are reduced in fp64 (two passes, no
// cancellation), then a third pass writes the normalized output.
#include <cuda_runtime.h>

constexpr int kThreads = 1024;
constexpr float kEps = 1e-5f;

__device__ double blockSumD(double v) {
    __shared__ double warp_sums[32];
    __shared__ double total;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x % 32 == 0) warp_sums[threadIdx.x / 32] = v;
    __syncthreads();
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

__global__ void layerNorm(const float* x, const float* gamma, const float* beta, float* y, size_t group) {
    const float* xg = x + blockIdx.x * group;
    float* yg = y + blockIdx.x * group;
    double s = 0.0;
    for (size_t i = threadIdx.x; i < group; i += kThreads) s += xg[i];
    const float mean = static_cast<float>(blockSumD(s) / group);
    double sq = 0.0;
    for (size_t i = threadIdx.x; i < group; i += kThreads) {
        const double t = xg[i] - mean;
        sq += t * t;
    }
    const float rstd = static_cast<float>(1.0 / sqrt(blockSumD(sq) / group + kEps));
    for (size_t i = threadIdx.x; i < group; i += kThreads) yg[i] = (xg[i] - mean) * rstd * gamma[i] + beta[i];
}

// X, gamma, beta, Y are device pointers
extern "C" void solution(const float* X, const float* gamma, const float* beta, float* Y, size_t B, size_t F, size_t D1, size_t D2) {
    layerNorm<<<static_cast<unsigned>(B), kThreads>>>(X, gamma, beta, Y, F * D1 * D2);
}

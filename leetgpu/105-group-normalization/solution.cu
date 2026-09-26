// Group Normalization (LeetGPU)
// https://leetgpu.com/challenges/group-normalization
//
// In NCHW layout the (C/G) x H x W elements of one (n, g) group are contiguous,
// so each group is one block-sized reduction:
//   pass 1: sum and sum of squares in fp64 (block reduction) -> mean, rstd;
//   pass 2: y = (x - mean) * rstd * gamma[c] + beta[c].
#include <cuda_runtime.h>

constexpr int kThreads = 256;

__device__ void blockSum2(double& a, double& b) {
    __shared__ double s_a[kThreads / 32];
    __shared__ double s_b[kThreads / 32];
    for (int offset = 16; offset > 0; offset >>= 1) {
        a += __shfl_xor_sync(0xffffffffu, a, offset);
        b += __shfl_xor_sync(0xffffffffu, b, offset);
    }
    if (threadIdx.x % 32 == 0) {
        s_a[threadIdx.x / 32] = a;
        s_b[threadIdx.x / 32] = b;
    }
    __syncthreads();
    a = 0.0;
    b = 0.0;
    for (int w = 0; w < kThreads / 32; ++w) {
        a += s_a[w];
        b += s_b[w];
    }
}

__global__ void groupNorm(const float* x, const float* gamma, const float* beta, float* y, int c, int hw, int groups,
                          float eps) {
    const int n = blockIdx.x / groups;
    const int g = blockIdx.x % groups;
    const int cpg = c / groups;
    const size_t count = static_cast<size_t>(cpg) * hw;
    const size_t base = (static_cast<size_t>(n) * c + static_cast<size_t>(g) * cpg) * hw;
    double s = 0.0, sq = 0.0;
    for (size_t i = threadIdx.x; i < count; i += kThreads) {
        const double v = x[base + i];
        s += v;
        sq += v * v;
    }
    blockSum2(s, sq);
    const double mean = s / count;
    const double var = sq / count - mean * mean;
    const float mean_f = static_cast<float>(mean);
    const float rstd = static_cast<float>(1.0 / sqrt((var > 0.0 ? var : 0.0) + eps));
    for (size_t i = threadIdx.x; i < count; i += kThreads) {
        const int ch = g * cpg + static_cast<int>(i / hw);
        y[base + i] = (x[base + i] - mean_f) * rstd * gamma[ch] + beta[ch];
    }
}

// X, gamma, beta, Y are device pointers
extern "C" void solve(const float* X, const float* gamma, const float* beta, float* Y, int N, int C, int H, int W, int G,
                      float eps) {
    groupNorm<<<N * G, kThreads>>>(X, gamma, beta, Y, C, H * W, G, eps);
    cudaDeviceSynchronize();
}

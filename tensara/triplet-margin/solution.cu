// Triplet Margin Loss (Tensara)
// https://tensara.org/problems/triplet-margin
//
// loss = mean_i max(||a_i - p_i + eps||_2 - ||a_i - n_i + eps||_2 + margin, 0),
// eps = 1e-6 (torch.pairwise_distance). One block per sample computes both
// distances in one pass; per-sample terms go to a small buffer that a final
// block averages in fp64.
#include <cuda_runtime.h>

constexpr int kThreads = 256;
constexpr float kPairEps = 1e-6f;

__device__ float blockSum(float v) {
    // Block-wide sum; every thread receives the result.
    __shared__ float warp_sums[32];
    __shared__ float total;
    // Butterfly sum inside each warp; lane 0 publishes the warp total.
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x % 32 == 0) warp_sums[threadIdx.x / 32] = v;
    __syncthreads();
    // Thread 0 adds the warp totals and broadcasts the result through shared memory.
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

__global__ void tripletTerms(const float* a, const float* p, const float* n, float* terms, size_t e, float margin) {
    // One block per triplet: both squared distances in a single pass (the anchor is read once).
    const size_t base = blockIdx.x * e;
    float dp = 0.0f, dn = 0.0f;
    for (size_t j = threadIdx.x; j < e; j += kThreads) {
        const float av = a[base + j];
        const float x = av - p[base + j] + kPairEps;
        const float y = av - n[base + j] + kPairEps;
        dp += x * x;
        dn += y * y;
    }
    // Reduce across the block; thread 0 stores the hinge term max(d(a,p) - d(a,n) + margin, 0).
    dp = blockSum(dp);
    dn = blockSum(dn);
    if (threadIdx.x == 0) terms[blockIdx.x] = fmaxf(sqrtf(dp) - sqrtf(dn) + margin, 0.0f);
}

__global__ void meanTerms(const float* terms, float* loss, size_t b) {
    // Mean of the B hinge terms in double (one block).
    double s = 0.0;
    for (size_t i = threadIdx.x; i < b; i += kThreads) s += terms[i];
    s = blockSumD(s);
    if (threadIdx.x == 0) loss[0] = static_cast<float>(s / b);
}

// anchor, positive, negative, loss are device pointers
extern "C" void solution(const float* anchor, const float* positive, const float* negative, float* loss, size_t B, size_t E, float margin) {
    // Temporary buffer for the per-triplet terms.
    float* terms = nullptr;
    cudaMalloc(&terms, B * sizeof(float));
    tripletTerms<<<static_cast<unsigned>(B), kThreads>>>(anchor, positive, negative, terms, E, margin);
    meanTerms<<<1, kThreads>>>(terms, loss, B);
    cudaDeviceSynchronize();
    cudaFree(terms);
}

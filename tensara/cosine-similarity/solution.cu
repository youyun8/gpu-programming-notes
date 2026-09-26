// Cosine Similarity Loss (Tensara)
// https://tensara.org/problems/cosine-similarity
//
// out[i] = 1 - (p_i . t_i) / max(||p_i|| ||t_i||, 1e-8) for each of n rows of
// length d. One block per row reduces the three dot products (p.t, p.p, t.t)
// together in a single pass.
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

__global__ void cosineRows(const float* p, const float* t, float* out, size_t d) {
    const float* pr = p + blockIdx.x * d;
    const float* tr = t + blockIdx.x * d;
    float dot = 0.0f, pp = 0.0f, tt = 0.0f;
    for (size_t j = threadIdx.x; j < d; j += kThreads) {
        const float a = pr[j], b = tr[j];
        dot += a * b;
        pp += a * a;
        tt += b * b;
    }
    dot = blockSum(dot);
    pp = blockSum(pp);
    tt = blockSum(tt);
    if (threadIdx.x == 0) out[blockIdx.x] = 1.0f - dot / sqrtf(fmaxf(pp * tt, 1e-16f));
}

// predictions, targets, output are device pointers
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n, size_t d) {
    cosineRows<<<static_cast<unsigned>(n), kThreads>>>(predictions, targets, output, d);
}

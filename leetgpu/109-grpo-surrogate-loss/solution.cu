// GRPO Surrogate Loss (LeetGPU)
// https://leetgpu.com/challenges/grpo-surrogate-loss
//
// 1. groupAdvantages: per prompt b, A[b, g] = (R - mean) / (std + 1e-8) over G
//    (population std), one thread per prompt (G is small).
// 2. tokenSums: over all B*G*S tokens, min(r A, clip(r) A) - beta k3(ref, pi)
//    with k3 = e^d - d - 1, d = log_ref - log_pi; fp64 block partials.
// 3. finalize: -mean.
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 1024;

__device__ double g_partials[kMaxBlocks];

__global__ void groupAdvantages(const float* rewards, float* adv, int b, int g) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b) return;
    const float* r = rewards + static_cast<size_t>(i) * g;
    double mean = 0.0;
    for (int j = 0; j < g; ++j) mean += r[j];
    mean /= g;
    double var = 0.0;
    for (int j = 0; j < g; ++j) var += (r[j] - mean) * (r[j] - mean);
    const float std_dev = static_cast<float>(sqrt(var / g));
    for (int j = 0; j < g; ++j) adv[static_cast<size_t>(i) * g + j] = (r[j] - static_cast<float>(mean)) / (std_dev + 1e-8f);
}

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

__global__ void tokenSums(const float* adv, const float* log_pi, const float* log_pi_old, const float* log_ref, long long n,
                          int s, float clip_eps, float beta) {
    float local = 0.0f;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const float a = adv[i / s];
        const float r = expf(log_pi[i] - log_pi_old[i]);
        const float rc = fminf(fmaxf(r, 1.0f - clip_eps), 1.0f + clip_eps);
        const float surrogate = fminf(r * a, rc * a);
        const float d = log_ref[i] - log_pi[i];
        local += surrogate - beta * (expf(d) - d - 1.0f);
    }
    const double t = blockReduceSum(local);
    if (threadIdx.x == 0) g_partials[blockIdx.x] = t;
}

__global__ void finalize(float* out, int num_partials, long long n) {
    double v = 0.0;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) v += g_partials[i];
    v = blockReduceSum(v);
    if (threadIdx.x == 0) out[0] = static_cast<float>(-v / n);
}

// all pointers are device pointers
extern "C" void solve(const float* rewards, const float* log_pi, const float* log_pi_old, const float* log_ref,
                      float* output, float clip_eps, float beta, int B, int G, int S) {
    float* adv = nullptr;
    cudaMalloc(&adv, static_cast<size_t>(B) * G * sizeof(float));
    groupAdvantages<<<(B + 127) / 128, 128>>>(rewards, adv, B, G);
    const long long n = static_cast<long long>(B) * G * S;
    long long blocks = (n + kBlockSize - 1) / kBlockSize;
    blocks = blocks > kMaxBlocks ? kMaxBlocks : blocks;
    tokenSums<<<static_cast<int>(blocks), kBlockSize>>>(adv, log_pi, log_pi_old, log_ref, n, S, clip_eps, beta);
    finalize<<<1, kBlockSize>>>(output, static_cast<int>(blocks), n);
    cudaDeviceSynchronize();
    cudaFree(adv);
}

// Top-p (Nucleus) Sampling (LeetGPU)
// https://leetgpu.com/challenges/top-p-sampling
//
// Single block (vocab <= 50,000):
//   1. softmax statistics (max, sum of exp) with block reductions;
//   2. find the nucleus WITHOUT sorting: probabilities are positive floats, so
//      their bit patterns are order-preserving uint32 keys. A 32-step bitwise
//      search finds the largest threshold key T with mass(prob >= T) >= p.
//      {prob >= T} is exactly the smallest top set reaching p (the prefix of
//      the descending sort that searchsorted + 1 selects);
//   3. draw u in [0, 1) from the seed (SplitMix64) and pick the nucleus token
//      where the running (renormalized) mass crosses u, scanning in index order.
// Note: the reference draws with torch.multinomial, whose RNG stream cannot be
// reproduced from CUDA C++, so only the distribution (not the exact token for a
// multi-token nucleus) matches.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kThreads = 1024;

__device__ float blockSum(float v, float* scratch) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < kThreads / 32; ++w) total += scratch[w];
    __syncthreads();
    return total;
}

__device__ float blockMax(float v, float* scratch) {
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = v;
    __syncthreads();
    float total = -FLT_MAX;
    for (int w = 0; w < kThreads / 32; ++w) total = fmaxf(total, scratch[w]);
    __syncthreads();
    return total;
}

__device__ unsigned long long splitMix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__global__ void topPSample(const float* logits, const float* p_ptr, const int* seed_ptr, int* sampled, int v) {
    __shared__ float scratch[32];
    __shared__ int chunk_hit;
    const float p = p_ptr[0];

    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < v; i += kThreads) local_max = fmaxf(local_max, logits[i]);
    const float mx = blockMax(local_max, scratch);
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < v; i += kThreads) local_sum += expf(logits[i] - mx);
    const float inv_total = 1.0f / blockSum(local_sum, scratch);

    // Largest T (bitwise, MSB first) such that sum(prob >= T) >= p.
    unsigned int threshold = 0u;
    for (int bit = 31; bit >= 0; --bit) {
        const unsigned int candidate = threshold | (1u << bit);
        float mass = 0.0f;
        for (int i = threadIdx.x; i < v; i += kThreads) {
            const float prob = expf(logits[i] - mx) * inv_total;
            if (__float_as_uint(prob) >= candidate) mass += prob;
        }
        if (blockSum(mass, scratch) >= p) threshold = candidate;
    }

    float nucleus = 0.0f;
    for (int i = threadIdx.x; i < v; i += kThreads) {
        const float prob = expf(logits[i] - mx) * inv_total;
        if (__float_as_uint(prob) >= threshold) nucleus += prob;
    }
    const float nucleus_mass = blockSum(nucleus, scratch);

    const unsigned long long bits = splitMix64(static_cast<unsigned long long>(seed_ptr[0]));
    const float target = static_cast<float>((bits >> 40) * (1.0 / 16777216.0)) * nucleus_mass;

    // Inverse CDF over nucleus tokens in index order, one 1024-token chunk at a time.
    float carry = 0.0f;
    int last_token = 0;
    if (threadIdx.x == 0) chunk_hit = 0x7fffffff;
    __syncthreads();
    for (int base = 0; base < v; base += kThreads) {
        const int i = base + threadIdx.x;
        float prob = 0.0f;
        if (i < v) {
            prob = expf(logits[i] - mx) * inv_total;
            if (__float_as_uint(prob) < threshold) prob = 0.0f;
        }
        // Inclusive block scan of prob (warp shuffles + warp totals).
        float incl = prob;
        const int lane = threadIdx.x % 32;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float other = __shfl_up_sync(0xffffffffu, incl, offset);
            if (lane >= offset) incl += other;
        }
        if (lane == 31) scratch[threadIdx.x / 32] = incl;
        __syncthreads();
        float warp_prefix = 0.0f;
        for (int w = 0; w < static_cast<int>(threadIdx.x / 32); ++w) warp_prefix += scratch[w];
        float chunk_total = 0.0f;
        for (int w = 0; w < kThreads / 32; ++w) chunk_total += scratch[w];
        incl += warp_prefix + carry;
        if (prob > 0.0f && incl > target) atomicMin(&chunk_hit, i);
        if (prob > 0.0f) last_token = i;
        __syncthreads();
        if (chunk_hit != 0x7fffffff) break;
        carry += chunk_total;
        __syncthreads();
    }
    // Fallback for rounding at the very end of the CDF: the last nucleus token.
    for (int offset = 16; offset > 0; offset >>= 1) last_token = max(last_token, __shfl_xor_sync(0xffffffffu, last_token, offset));
    if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = __int_as_float(last_token);
    __syncthreads();
    if (threadIdx.x == 0) {
        int fallback = 0;
        for (int w = 0; w < kThreads / 32; ++w) fallback = max(fallback, __float_as_int(scratch[w]));
        sampled[0] = chunk_hit != 0x7fffffff ? chunk_hit : fallback;
    }
}

// logits, p, seed, sampled_token are device pointers
extern "C" void solve(const float* logits, const float* p, const int* seed, int* sampled_token, int vocab_size) {
    topPSample<<<1, kThreads>>>(logits, p, seed, sampled_token, vocab_size);
    cudaDeviceSynchronize();
}

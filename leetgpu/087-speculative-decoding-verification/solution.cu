// Speculative Decoding Verification (LeetGPU)
// https://leetgpu.com/challenges/speculative-decoding-verification
//
// One block per sequence. Positions are processed left to right (the loop
// stops at the first rejection), and the vocabulary-sized work - the residual
// distribution max(0, q - p), its normalization and the inverse-CDF search -
// is spread over the block with block scans.
#include <cuda_runtime.h>

constexpr int kThreads = 1024;

__device__ float blockSum(float v, float* scratch) {
    // Block-wide sum returned to every thread.
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = v;
    __syncthreads();
    float t = 0.0f;
    for (int w = 0; w < kThreads / 32; ++w) t += scratch[w];
    __syncthreads();
    return t;
}

// Smallest index with cdf[index] >= r where cdf is the running sum of
// weight(v) (v = 0..V-1), i.e. torch.searchsorted(cdf, r); V if none.
template <class WeightFn>
__device__ int inverseCdf(int vocab, float r, WeightFn weight, float* scratch, int* s_found) {
    // Scan the weights 1024 at a time (warp scans + warp totals + running carry); the first
    // index whose running sum reaches r wins (atomicMin), and the loop stops at that chunk.
    if (threadIdx.x == 0) *s_found = vocab;
    __syncthreads();
    float carry = 0.0f;
    const int lane = threadIdx.x % 32;
    for (int base = 0; base < vocab; base += kThreads) {
        const int v = base + threadIdx.x;
        const float wv = v < vocab ? weight(v) : 0.0f;
        float incl = wv;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float other = __shfl_up_sync(0xffffffffu, incl, offset);
            if (lane >= offset) incl += other;
        }
        if (lane == 31) scratch[threadIdx.x / 32] = incl;
        __syncthreads();
        float before = carry;
        for (int w = 0; w < static_cast<int>(threadIdx.x / 32); ++w) before += scratch[w];
        float chunk = 0.0f;
        for (int w = 0; w < kThreads / 32; ++w) chunk += scratch[w];
        if (v < vocab && before + incl >= r) atomicMin(s_found, v);
        __syncthreads();
        if (*s_found != vocab) break;
        carry += chunk;
        __syncthreads();
    }
    const int found = *s_found;
    __syncthreads();
    return found;
}

__global__ void verify(const int* draft, const float* p_draft, const float* p_target, const float* u, int* out, int t_len,
                       int vocab) {
    __shared__ float scratch[32];
    __shared__ int s_found;
    // One block per sequence: clear its output row; u[..., T] is the resampling draw.
    const int b = blockIdx.x;
    for (int i = threadIdx.x; i <= t_len; i += kThreads) out[static_cast<size_t>(b) * (t_len + 1) + i] = 0;
    __syncthreads();
    const float r = u[static_cast<size_t>(b) * (t_len + 1) + t_len];

    // Walk the draft tokens in order: accept token i with probability min(1, q/p).
    for (int i = 0; i < t_len; ++i) {
        const int tok = draft[static_cast<size_t>(b) * t_len + i];
        const float* p = p_draft + (static_cast<size_t>(b) * t_len + i) * vocab;
        const float* q = p_target + (static_cast<size_t>(b) * t_len + i) * vocab;
        const float alpha = fminf(1.0f, q[tok] / p[tok]);
        if (u[static_cast<size_t>(b) * (t_len + 1) + i] < alpha) {
            if (threadIdx.x == 0) out[static_cast<size_t>(b) * (t_len + 1) + i] = tok;
            continue;
        }
        // Rejected: resample from normalize(max(0, q - p)), uniform if it is all zero.
        float local = 0.0f;
        for (int v = threadIdx.x; v < vocab; v += kThreads) local += fmaxf(q[v] - p[v], 0.0f);
        const float total = blockSum(local, scratch);
        int new_tok;
        if (total > 0.0f) {
            new_tok = inverseCdf(vocab, r, [&](int v) { return fmaxf(q[v] - p[v], 0.0f) / total; }, scratch, &s_found);
        } else {
            new_tok = inverseCdf(vocab, r, [&](int) { return 1.0f / vocab; }, scratch, &s_found);
        }
        if (threadIdx.x == 0) out[static_cast<size_t>(b) * (t_len + 1) + i] = min(new_tok, vocab - 1);
        return;
    }
    // All draft tokens accepted: bonus token from the target distribution at the last position.
    const float* q_last = p_target + (static_cast<size_t>(b) * t_len + t_len - 1) * vocab;
    const int bonus = inverseCdf(vocab, r, [&](int v) { return q_last[v]; }, scratch, &s_found);
    if (threadIdx.x == 0) out[static_cast<size_t>(b) * (t_len + 1) + t_len] = min(bonus, vocab - 1);
}

// draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens are device pointers
extern "C" void solve(const int* draft_tokens, const float* draft_probs, const float* target_probs,
                      const float* uniform_samples, int* output_tokens, int B, int T, int V) {
    verify<<<B, kThreads>>>(draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens, T, V);
    cudaDeviceSynchronize();
}

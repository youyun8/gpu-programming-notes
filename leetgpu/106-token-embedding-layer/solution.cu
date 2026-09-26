// Token Embedding Layer (LeetGPU)
// https://leetgpu.com/challenges/token-embedding-layer
//
// y = LayerNorm(E_tok[token] + E_pos[position]) * gamma + beta, D <= 1024.
// One warp per token: gather both rows (coalesced), keep the D/32 <= 32 sums
// per lane in registers, compute mean and then the centered variance from the
// registers (two-pass, no cancellation), normalize and store - the embedding
// sum is never written to memory.
#include <cuda_runtime.h>

constexpr int kWarpsPerBlock = 8;
constexpr int kMaxPerLane = 32;  // D <= 1024

__global__ void embedLayerNorm(const int* token_ids, const int* position_ids, const float* tok_emb, const float* pos_emb,
                               const float* gamma, const float* beta, float* out, int bt, int t_len, int d, float eps) {
    // One warp per token; lane l holds columns l, l + 32, ... of the row in registers.
    const int lane = threadIdx.x % 32;
    const int token = blockIdx.x * kWarpsPerBlock + threadIdx.x / 32;
    if (token >= bt) return;
    // Gather the token and position embedding rows and add them.
    const float* tr = tok_emb + static_cast<size_t>(token_ids[token]) * d;
    const float* pr = pos_emb + static_cast<size_t>(position_ids[token % t_len]) * d;
    float vals[kMaxPerLane];
    float sum = 0.0f;
#pragma unroll
    for (int r = 0; r < kMaxPerLane; ++r) {
        const int c = lane + 32 * r;
        vals[r] = c < d ? tr[c] + pr[c] : 0.0f;
        sum += vals[r];
    }
    // LayerNorm: mean by a butterfly sum...
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    const float mean = sum / d;
    // ...then the centered variance (two passes over registers, no cancellation)...
    float sq = 0.0f;
#pragma unroll
    for (int r = 0; r < kMaxPerLane; ++r) {
        const int c = lane + 32 * r;
        if (c < d) {
            const float diff = vals[r] - mean;
            sq += diff * diff;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, offset);
    const float rstd = rsqrtf(sq / d + eps);
    // ...and the affine output.
    float* o = out + static_cast<size_t>(token) * d;
#pragma unroll
    for (int r = 0; r < kMaxPerLane; ++r) {
        const int c = lane + 32 * r;
        if (c < d) o[c] = (vals[r] - mean) * rstd * gamma[c] + beta[c];
    }
}

// all pointers are device pointers
extern "C" void solve(const int* token_ids, const int* position_ids, const float* token_embeddings,
                      const float* position_embeddings, const float* gamma, const float* beta, float* output, int B,
                      int T, int V, int P, int D, float eps) {
    const int bt = B * T;
    embedLayerNorm<<<(bt + kWarpsPerBlock - 1) / kWarpsPerBlock, kWarpsPerBlock * 32>>>(
        token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta, output, bt, T, D, eps);
    cudaDeviceSynchronize();
}

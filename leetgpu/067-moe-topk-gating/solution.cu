// MoE Top-K Gating (LeetGPU)
// https://leetgpu.com/challenges/moe-top-k-gating
//
// Per token: the k largest of E <= 256 logits (descending, lower index first on
// ties like torch.topk), then softmax over those k values.
// One warp per token: each lane holds E/32 <= 8 logits in registers; k rounds
// of a warp arg-max (value, index) pick the winners, marking them as used.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kWarpsPerBlock = 8;
constexpr int kMaxPerLane = 8;  // E <= 256

__global__ void moeGating(const float* logits, float* weights, int* indices, int m, int e, int k) {
    // One warp per token row; lane l holds experts l, l + 32, ... (up to 8 per lane) in registers.
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * kWarpsPerBlock + threadIdx.x / 32;
    if (row >= m) return;
    const float* z = logits + static_cast<size_t>(row) * e;
    float vals[kMaxPerLane];
#pragma unroll
    for (int r = 0; r < kMaxPerLane; ++r) {
        const int j = lane + 32 * r;
        vals[r] = j < e ? z[j] : -FLT_MAX;
    }
    // k rounds of warp-wide arg-max, excluding experts already picked.
    unsigned int used = 0;  // bit r: vals[r] already selected
    float top_val = 0.0f;
    float sum = 0.0f;
    for (int t = 0; t < k; ++t) {
        // Each lane's best unused expert...
        float best = -FLT_MAX;
        int best_idx = 0x7fffffff;
#pragma unroll
        for (int r = 0; r < kMaxPerLane; ++r) {
            const int j = lane + 32 * r;
            if (j < e && !((used >> r) & 1u) && (vals[r] > best || best_idx == 0x7fffffff)) {
                best = vals[r];
                best_idx = j;
            }
        }
        // ...then a butterfly arg-max across the warp (ties -> lower expert index).
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, offset);
            const int oi = __shfl_xor_sync(0xffffffffu, best_idx, offset);
            if (ov > best || (ov == best && oi < best_idx)) {
                best = ov;
                best_idx = oi;
            }
        }
        // The owning lane marks the winner as used.
        if (best_idx % 32 == lane) used |= 1u << (best_idx / 32);
        if (t == 0) top_val = best;  // the first winner is the max: softmax shift
        // Softmax over the k selected logits: unnormalized weight now, normalized after the loop.
        const float ex = expf(best - top_val);
        sum += ex;
        if (lane == 0) {
            weights[static_cast<size_t>(row) * k + t] = ex;
            indices[static_cast<size_t>(row) * k + t] = best_idx;
        }
    }
    __syncwarp();
    // Normalize the k weights.
    const float inv = 1.0f / sum;
    for (int t = lane; t < k; t += 32) weights[static_cast<size_t>(row) * k + t] *= inv;
}

// logits, topk_weights, topk_indices are device pointers
extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E, int k) {
    moeGating<<<(M + kWarpsPerBlock - 1) / kWarpsPerBlock, kWarpsPerBlock * 32>>>(logits, topk_weights, topk_indices, M, E, k);
    cudaDeviceSynchronize();
}

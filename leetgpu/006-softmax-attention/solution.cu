// Softmax Attention (LeetGPU)
// https://leetgpu.com/challenges/softmax-attention
//
// output = softmax(Q K^T / sqrt(d)) V, Q: M x d, K/V: N x d, d <= 128.
//
// FlashAttention-style single pass, never materializing the M x N score matrix:
// - one warp per query row, 4 warps per block share K/V tiles of 32 keys that
//   are staged in shared memory;
// - lane l scores key l of the tile (full dot product over d), so a tile costs
//   one warp max + one warp sum instead of a reduction per key;
// - the running max / denominator are rescaled per tile (online softmax) and
//   each lane accumulates output columns lane, lane+32, lane+64, lane+96.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kWarps = 4;
constexpr int kTileKeys = 32;
constexpr int kMaxDim = 128;
constexpr int kPitch = kMaxDim + 1;  // odd pitch: per-lane row reads hit distinct banks
constexpr int kColsPerLane = kMaxDim / 32;

// Butterfly reductions: every lane receives the result.
__device__ __forceinline__ float warpMax(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    return v;
}

__device__ __forceinline__ float warpSum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

__global__ void __launch_bounds__(kWarps * 32)
attentionKernel(const float* q, const float* k, const float* v, float* output, int m, int n, int d, float scale) {
    // Shared tiles of 32 keys and values, and the 4 query rows of this block.
    __shared__ float k_tile[kTileKeys][kPitch];
    __shared__ float v_tile[kTileKeys][kMaxDim];
    __shared__ float q_rows[kWarps][kMaxDim];

    // One warp per query row, 4 rows per block.
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int row = blockIdx.x * kWarps + warp;
    const bool active = row < m;

    // Stage the query row, pre-multiplied by 1/sqrt(d).
    for (int c = lane; c < d; c += 32) q_rows[warp][c] = active ? q[static_cast<size_t>(row) * d + c] * scale : 0.0f;

    // Output accumulator (lane l owns columns l, l + 32, ...) and online-softmax state.
    float acc[kColsPerLane] = {};
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    // Stream K and V in tiles of 32 keys.
    for (int key0 = 0; key0 < n; key0 += kTileKeys) {
        __syncthreads();  // previous tile fully consumed (and q_rows visible on first pass)
        // Stage the tile (zero rows past the last key).
        for (int i = threadIdx.x; i < kTileKeys * d; i += blockDim.x) {
            const int j = i / d;
            const int c = i % d;
            const bool valid = key0 + j < n;
            const size_t g = static_cast<size_t>(key0 + j) * d + c;
            k_tile[j][c] = valid ? k[g] : 0.0f;
            v_tile[j][c] = valid ? v[g] : 0.0f;
        }
        __syncthreads();

        // Lane l scores key l of the tile.
        float score = -FLT_MAX;
        if (key0 + lane < n) {
            score = 0.0f;
            for (int c = 0; c < d; ++c) score = fmaf(q_rows[warp][c], k_tile[lane][c], score);
        }
        // Online softmax update: new max, rescale factor, probabilities, running sum.
        const float new_max = fmaxf(running_max, warpMax(score));
        const float correction = expf(running_max - new_max);
        const float p = key0 + lane < n ? expf(score - new_max) : 0.0f;
        running_sum = running_sum * correction + warpSum(p);
        running_max = new_max;

        // Rescale the accumulator, then add P V; key j's probability is broadcast from lane j.
#pragma unroll
        for (int r = 0; r < kColsPerLane; ++r) acc[r] *= correction;
        const int tile_keys = min(kTileKeys, n - key0);
        for (int j = 0; j < tile_keys; ++j) {
            const float pj = __shfl_sync(0xffffffffu, p, j);
#pragma unroll
            for (int r = 0; r < kColsPerLane; ++r) {
                const int c = lane + 32 * r;
                if (c < d) acc[r] = fmaf(pj, v_tile[j][c], acc[r]);
            }
        }
    }

    // Normalize and store the output row.
    if (active) {
        const float inv_sum = 1.0f / running_sum;
#pragma unroll
        for (int r = 0; r < kColsPerLane; ++r) {
            const int c = lane + 32 * r;
            if (c < d) output[static_cast<size_t>(row) * d + c] = acc[r] * inv_sum;
        }
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    // One block per 4 query rows.
    const int num_blocks = (M + kWarps - 1) / kWarps;
    attentionKernel<<<num_blocks, kWarps * 32>>>(Q, K, V, output, M, N, d, 1.0f / sqrtf(static_cast<float>(d)));
    cudaDeviceSynchronize();
}

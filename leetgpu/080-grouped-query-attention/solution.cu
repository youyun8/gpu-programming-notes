// Grouped Query Attention (LeetGPU)
// https://leetgpu.com/challenges/grouped-query-attention
//
// Q: (Hq, S, D), K/V: (Hkv, S, D); query head h uses KV head h / (Hq / Hkv).
// FlashAttention-style single pass (no S x S score matrix):
//   - a block = 8 warps = 8 consecutive query rows of one query head; the
//     block streams its KV head through shared memory in tiles of 32 keys;
//   - lane l scores key l of the tile, so a tile needs one warp max and one
//     warp sum; the running max / denominator are rescaled per tile;
//   - lane l accumulates output columns l, l + 32, ... (D <= 256).
// Shared memory scales with D (up to ~74 KB), so it is dynamic and opted in.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kWarps = 8;
constexpr int kTileKeys = 32;
constexpr int kMaxColsPerLane = 8;  // head_dim <= 256

__global__ void __launch_bounds__(kWarps * 32)
gqaKernel(const float* q, const float* k, const float* v, float* out, int group, int seq, int d, float scale) {
    extern __shared__ float smem[];
    const int pitch = d + 1;  // odd pitch -> per-lane K row reads hit distinct banks
    float* k_tile = smem;                          // [32][d + 1]
    float* v_tile = k_tile + kTileKeys * pitch;    // [32][d]
    float* q_rows = v_tile + kTileKeys * d;        // [kWarps][d]

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int head = blockIdx.y;
    const int kv_head = head / group;
    const int row = blockIdx.x * kWarps + warp;
    const bool active = row < seq;

    const float* q_row = q + (static_cast<size_t>(head) * seq + row) * d;
    const float* k_head = k + static_cast<size_t>(kv_head) * seq * d;
    const float* v_head = v + static_cast<size_t>(kv_head) * seq * d;
    for (int c = lane; c < d; c += 32) q_rows[warp * d + c] = active ? q_row[c] * scale : 0.0f;

    float acc[kMaxColsPerLane] = {};
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    for (int key0 = 0; key0 < seq; key0 += kTileKeys) {
        __syncthreads();
        for (int i = threadIdx.x; i < kTileKeys * d; i += blockDim.x) {
            const int j = i / d;
            const int c = i % d;
            const bool valid = key0 + j < seq;
            const size_t g = static_cast<size_t>(key0 + j) * d + c;
            k_tile[j * pitch + c] = valid ? k_head[g] : 0.0f;
            v_tile[j * d + c] = valid ? v_head[g] : 0.0f;
        }
        __syncthreads();

        float score = -FLT_MAX;
        if (key0 + lane < seq) {
            score = 0.0f;
            for (int c = 0; c < d; ++c) score = fmaf(q_rows[warp * d + c], k_tile[lane * pitch + c], score);
        }
        float tile_max = score;
        for (int offset = 16; offset > 0; offset >>= 1) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset));
        const float new_max = fmaxf(running_max, tile_max);
        const float correction = expf(running_max - new_max);
        const float p = key0 + lane < seq ? expf(score - new_max) : 0.0f;
        float tile_sum = p;
        for (int offset = 16; offset > 0; offset >>= 1) tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, offset);
        running_sum = running_sum * correction + tile_sum;
        running_max = new_max;
#pragma unroll
        for (int r = 0; r < kMaxColsPerLane; ++r) acc[r] *= correction;
        const int tile_keys = min(kTileKeys, seq - key0);
        for (int j = 0; j < tile_keys; ++j) {
            const float pj = __shfl_sync(0xffffffffu, p, j);
#pragma unroll
            for (int r = 0; r < kMaxColsPerLane; ++r) {
                const int c = lane + 32 * r;
                if (c < d) acc[r] = fmaf(pj, v_tile[j * d + c], acc[r]);
            }
        }
    }
    if (active) {
        const float inv = 1.0f / running_sum;
        float* o = out + (static_cast<size_t>(head) * seq + row) * d;
#pragma unroll
        for (int r = 0; r < kMaxColsPerLane; ++r) {
            const int c = lane + 32 * r;
            if (c < d) o[c] = acc[r] * inv;
        }
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int num_q_heads, int num_kv_heads,
                      int seq_len, int head_dim) {
    const int d = head_dim;
    const size_t smem = (static_cast<size_t>(kTileKeys) * (d + 1) + static_cast<size_t>(kTileKeys) * d +
                         static_cast<size_t>(kWarps) * d) * sizeof(float);
    cudaFuncSetAttribute(gqaKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    const dim3 grid((seq_len + kWarps - 1) / kWarps, num_q_heads);
    gqaKernel<<<grid, kWarps * 32, smem>>>(Q, K, V, output, num_q_heads / num_kv_heads, seq_len, d,
                                           1.0f / sqrtf(static_cast<float>(d)));
    cudaDeviceSynchronize();
}

// Decaying Causal Attention (RetNet parallel form) (LeetGPU)
// https://leetgpu.com/challenges/decaying-causal-attention
//
// out[n] = sum_{m <= n} gamma^(n - m) * (Q[n] . K[m] / sqrt(d)) * V[m], d <= 256.
// No softmax, so no running max / renormalization - just a masked, weighted
// sum. Structure follows the flash kernels: 8 warps = 8 query rows per block,
// K/V tiles of 32 keys in (dynamic) shared memory, lane l scores key l, then
// each lane accumulates its output columns with shuffled weights. Key tiles
// stop at the block's last row (causality halves the work).
#include <cuda_runtime.h>

constexpr int kWarps = 8;
constexpr int kTileKeys = 32;
constexpr int kMaxColsPerLane = 8;  // d <= 256

__global__ void __launch_bounds__(kWarps * 32)
retention(const float* q, const float* k, const float* v, float* out, int seq, int d, float gamma, float scale) {
    extern __shared__ float smem[];
    const int pitch = d + 1;
    float* k_tile = smem;
    float* v_tile = k_tile + kTileKeys * pitch;
    float* q_rows = v_tile + kTileKeys * d;

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int row = blockIdx.x * kWarps + warp;
    const bool active = row < seq;
    for (int c = lane; c < d; c += 32) q_rows[warp * d + c] = active ? q[static_cast<size_t>(row) * d + c] * scale : 0.0f;

    float acc[kMaxColsPerLane] = {};
    const int last_row = min(seq - 1, blockIdx.x * kWarps + kWarps - 1);
    for (int key0 = 0; key0 <= last_row; key0 += kTileKeys) {
        __syncthreads();
        for (int i = threadIdx.x; i < kTileKeys * d; i += blockDim.x) {
            const int j = i / d;
            const int c = i % d;
            const bool valid = key0 + j < seq;
            const size_t g = static_cast<size_t>(key0 + j) * d + c;
            k_tile[j * pitch + c] = valid ? k[g] : 0.0f;
            v_tile[j * d + c] = valid ? v[g] : 0.0f;
        }
        __syncthreads();
        const int key = key0 + lane;
        float weight = 0.0f;
        if (active && key <= row) {
            float s = 0.0f;
            for (int c = 0; c < d; ++c) s = fmaf(q_rows[warp * d + c], k_tile[lane * pitch + c], s);
            weight = s * powf(gamma, static_cast<float>(row - key));
        }
        const int tile_keys = min(kTileKeys, seq - key0);
        for (int j = 0; j < tile_keys; ++j) {
            const float wj = __shfl_sync(0xffffffffu, weight, j);
#pragma unroll
            for (int r = 0; r < kMaxColsPerLane; ++r) {
                const int c = lane + 32 * r;
                if (c < d) acc[r] = fmaf(wj, v_tile[j * d + c], acc[r]);
            }
        }
    }
    if (active) {
#pragma unroll
        for (int r = 0; r < kMaxColsPerLane; ++r) {
            const int c = lane + 32 * r;
            if (c < d) out[static_cast<size_t>(row) * d + c] = acc[r];
        }
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int seq_len, int d_model, float gamma) {
    const int d = d_model;
    const size_t smem = (static_cast<size_t>(kTileKeys) * (2 * d + 1) + static_cast<size_t>(kWarps) * d) * sizeof(float);
    cudaFuncSetAttribute(retention, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    retention<<<(seq_len + kWarps - 1) / kWarps, kWarps * 32, smem>>>(Q, K, V, output, seq_len, d, gamma,
                                                                      1.0f / sqrtf(static_cast<float>(d)));
    cudaDeviceSynchronize();
}

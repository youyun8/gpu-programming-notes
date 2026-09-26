// Causal Attention (LeetGPU)
// https://leetgpu.com/challenges/causal-self-attention
//
// softmax(mask(Q K^T / sqrt(d))) V with key j visible to query i iff j <= i.
// Flash-style kernel: 8 warps = 8 query rows per block, 32-key tiles in shared
// memory, lane-per-key scoring, online softmax. Key tiles stop at the block's
// last row, which halves the work compared with dense attention.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kWarps = 8;
constexpr int kTile = 32;
constexpr int kMaxDim = 128;
constexpr int kPitch = kMaxDim + 1;
constexpr int kCols = kMaxDim / 32;

__global__ void causalAttention(const float* q, const float* k, const float* v, float* out, int m, int d, float scale) {
    __shared__ float k_t[kTile][kPitch];
    __shared__ float v_t[kTile][kPitch];
    __shared__ float q_s[kWarps][kMaxDim];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int row = blockIdx.x * kWarps + warp;
    const bool active = row < m;
    const int last_row = min(m - 1, blockIdx.x * kWarps + kWarps - 1);
    for (int c = lane; c < d; c += 32) q_s[warp][c] = active ? q[static_cast<size_t>(row) * d + c] * scale : 0.0f;
    float acc[kCols] = {};
    float mx = -FLT_MAX, sum = 0.0f;
    for (int j0 = 0; j0 <= last_row; j0 += kTile) {
        __syncthreads();
        for (int i = threadIdx.x; i < kTile * d; i += blockDim.x) {
            const int r = i / d, c = i % d;
            const bool ok = j0 + r < m;
            const size_t g = static_cast<size_t>(j0 + r) * d + c;
            k_t[r][c] = ok ? k[g] : 0.0f;
            v_t[r][c] = ok ? v[g] : 0.0f;
        }
        __syncthreads();
        const bool allowed = active && j0 + lane <= row;
        float s = -FLT_MAX;
        if (allowed) {
            s = 0.0f;
            for (int c = 0; c < d; ++c) s = fmaf(q_s[warp][c], k_t[lane][c], s);
        }
        float tile_max = s;
        for (int o = 16; o > 0; o >>= 1) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, o));
        const float new_mx = fmaxf(mx, tile_max);
        const float corr = expf(mx - new_mx);
        const float p = allowed ? expf(s - new_mx) : 0.0f;
        float tile_sum = p;
        for (int o = 16; o > 0; o >>= 1) tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, o);
        sum = sum * corr + tile_sum;
        mx = new_mx;
        for (int r = 0; r < kCols; ++r) acc[r] *= corr;
        for (int j = 0; j < min(kTile, m - j0); ++j) {
            const float pj = __shfl_sync(0xffffffffu, p, j);
            for (int r = 0; r < kCols; ++r) {
                const int c = lane + 32 * r;
                if (c < d) acc[r] = fmaf(pj, v_t[j][c], acc[r]);
            }
        }
    }
    if (active) {
        const float inv = 1.0f / sum;
        for (int r = 0; r < kCols; ++r) {
            const int c = lane + 32 * r;
            if (c < d) out[static_cast<size_t>(row) * d + c] = acc[r] * inv;
        }
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    causalAttention<<<(M + kWarps - 1) / kWarps, kWarps * 32>>>(Q, K, V, output, M, d, 1.0f / sqrtf(static_cast<float>(d)));
    cudaDeviceSynchronize();
}

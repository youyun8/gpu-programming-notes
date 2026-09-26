// Sliding Window Attention (LeetGPU)
// https://leetgpu.com/challenges/sliding-window-self-attention
//
// Query i attends to keys j with |i - j| <= w. Flash-style kernel (warp per
// query row, 32-key shared tiles, online softmax) that only visits the key
// range [first_row - w, last_row + w] of its 8 rows and masks per lane:
// O(M * w) work instead of O(M^2).
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kWarps = 8;
constexpr int kTile = 32;
constexpr int kMaxDim = 128;
constexpr int kPitch = kMaxDim + 1;
constexpr int kCols = kMaxDim / 32;

__global__ void windowAttention(const float* q, const float* k, const float* v, float* out, int m, int d, int w, float scale) {
    // Shared tiles of 32 keys and values (odd pitch: conflict-free per-lane rows) and the 8 query rows.
    __shared__ float k_t[kTile][kPitch];
    __shared__ float v_t[kTile][kPitch];
    __shared__ float q_s[kWarps][kMaxDim];
    // One warp per query row, 8 rows per block. Only keys in [first_row - w, last_row + w]
    // can be inside some row's window, so the key loop covers just that range.
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int first_row = blockIdx.x * kWarps;
    const int row = first_row + warp;
    const bool active = row < m;
    const int last_row = min(m - 1, first_row + kWarps - 1);
    const int lo = max(0, first_row - w);
    const int hi = min(m - 1, last_row + w);
    // Stage the query row, pre-scaled by 1/sqrt(d); set up the accumulator and softmax state.
    for (int c = lane; c < d; c += 32) q_s[warp][c] = active ? q[static_cast<size_t>(row) * d + c] * scale : 0.0f;
    float acc[kCols] = {};
    float mx = -FLT_MAX, sum = 0.0f;
    // Stream the needed keys and values in tiles of 32.
    for (int j0 = lo; j0 <= hi; j0 += kTile) {
        __syncthreads();
        for (int i = threadIdx.x; i < kTile * d; i += blockDim.x) {
            const int r = i / d, c = i % d;
            const bool ok = j0 + r <= hi;
            const size_t g = static_cast<size_t>(j0 + r) * d + c;
            k_t[r][c] = ok ? k[g] : 0.0f;
            v_t[r][c] = ok ? v[g] : 0.0f;
        }
        __syncthreads();
        // Lane l scores key j0 + l, masked unless |j - row| <= w.
        const int j = j0 + lane;
        const bool allowed = active && j <= hi && abs(j - row) <= w;
        float s = -FLT_MAX;
        if (allowed) {
            s = 0.0f;
            for (int c = 0; c < d; ++c) s = fmaf(q_s[warp][c], k_t[lane][c], s);
        }
        // Online softmax update: tile max, rescale factor, probabilities, running sum.
        float tile_max = s;
        for (int o = 16; o > 0; o >>= 1) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, o));
        const float new_mx = fmaxf(mx, tile_max);
        const float corr = expf(mx - new_mx);
        const float p = allowed ? expf(s - new_mx) : 0.0f;
        float tile_sum = p;
        for (int o = 16; o > 0; o >>= 1) tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, o);
        sum = sum * corr + tile_sum;
        mx = new_mx;
        // Rescale the accumulator, then add P V (probability of key jj broadcast from lane jj).
        for (int r = 0; r < kCols; ++r) acc[r] *= corr;
        for (int jj = 0; jj < min(kTile, hi - j0 + 1); ++jj) {
            const float pj = __shfl_sync(0xffffffffu, p, jj);
            for (int r = 0; r < kCols; ++r) {
                const int c = lane + 32 * r;
                if (c < d) acc[r] = fmaf(pj, v_t[jj][c], acc[r]);
            }
        }
    }
    // Normalize and store the output row.
    if (active) {
        const float inv = 1.0f / sum;
        for (int r = 0; r < kCols; ++r) {
            const int c = lane + 32 * r;
            if (c < d) out[static_cast<size_t>(row) * d + c] = acc[r] * inv;
        }
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d, int window_size) {
    windowAttention<<<(M + kWarps - 1) / kWarps, kWarps * 32>>>(Q, K, V, output, M, d, window_size,
                                                               1.0f / sqrtf(static_cast<float>(d)));
    cudaDeviceSynchronize();
}

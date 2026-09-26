// Attention with Sinks (StreamingLLM) (LeetGPU)
// https://leetgpu.com/challenges/attention-with-sinks
//
// Query i attends to keys j <= i with j < num_sinks (sinks) or j >= i - w + 1
// (sliding window). Flash-style kernel (warp per query row, 32-key tiles in
// shared memory, online softmax) that only visits the key tiles that can be
// allowed for the block's rows: [0, num_sinks) and [first_row - w + 1, last_row].
// Everything else is masked per lane. Work is O(M * (sinks + w)) instead of O(M^2).
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kWarps = 8;
constexpr int kTile = 32;
constexpr int kMaxDim = 128;
constexpr int kPitch = kMaxDim + 1;
constexpr int kCols = kMaxDim / 32;

__global__ void sinkAttention(const float* q, const float* k, const float* v, float* out, int m, int d, int sinks, int window,
                              float scale) {
    __shared__ float k_t[kTile][kPitch];
    __shared__ float v_t[kTile][kPitch];
    __shared__ float q_s[kWarps][kMaxDim];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int first_row = blockIdx.x * kWarps;
    const int last_row = min(m - 1, first_row + kWarps - 1);
    const int row = first_row + warp;
    const bool active = row < m;
    for (int c = lane; c < d; c += 32) q_s[warp][c] = active ? q[static_cast<size_t>(row) * d + c] * scale : 0.0f;

    float acc[kCols] = {};
    float mx = -FLT_MAX, sum = 0.0f;
    const int win_lo = max(first_row - window + 1, 0);
    // Two key ranges; the second starts after the first to avoid visiting a tile twice.
    const int range_end0 = min(sinks, last_row + 1);
    const int range_start1 = max(win_lo, range_end0);
    for (int pass = 0; pass < 2; ++pass) {
        const int begin = pass == 0 ? 0 : range_start1;
        const int end = pass == 0 ? range_end0 : last_row + 1;
        for (int j0 = begin; j0 < end; j0 += kTile) {
            __syncthreads();
            for (int i = threadIdx.x; i < kTile * d; i += blockDim.x) {
                const int r = i / d, c = i % d;
                const bool ok = j0 + r < end;
                const size_t g = static_cast<size_t>(j0 + r) * d + c;
                k_t[r][c] = ok ? k[g] : 0.0f;
                v_t[r][c] = ok ? v[g] : 0.0f;
            }
            __syncthreads();
            const int j = j0 + lane;
            const bool allowed = active && j < end && j <= row && (j < sinks || j >= row - window + 1);
            float s = -FLT_MAX;
            if (allowed) {
                s = 0.0f;
                for (int c = 0; c < d; ++c) s = fmaf(q_s[warp][c], k_t[lane][c], s);
            }
            float tile_max = s;
            for (int o = 16; o > 0; o >>= 1) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, o));
            const float new_mx = fmaxf(mx, tile_max);
            const float corr = (mx == -FLT_MAX) ? 0.0f : expf(mx - new_mx);
            const float p = allowed ? expf(s - new_mx) : 0.0f;
            float tile_sum = p;
            for (int o = 16; o > 0; o >>= 1) tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, o);
            sum = sum * corr + tile_sum;
            mx = new_mx;
            for (int r = 0; r < kCols; ++r) acc[r] *= corr;
            for (int jj = 0; jj < min(kTile, end - j0); ++jj) {
                const float pj = __shfl_sync(0xffffffffu, p, jj);
                for (int r = 0; r < kCols; ++r) {
                    const int c = lane + 32 * r;
                    if (c < d) acc[r] = fmaf(pj, v_t[jj][c], acc[r]);
                }
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
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d, int num_sinks,
                      int window_size) {
    sinkAttention<<<(M + kWarps - 1) / kWarps, kWarps * 32>>>(Q, K, V, output, M, d, num_sinks, window_size,
                                                             1.0f / sqrtf(static_cast<float>(d)));
    cudaDeviceSynchronize();
}

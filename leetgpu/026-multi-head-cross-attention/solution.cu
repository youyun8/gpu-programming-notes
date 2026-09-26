// Multi-Head Cross-Attention (LeetGPU)
// https://leetgpu.com/challenges/multi-head-cross-attention
//
// Q: (M, H, D), K/V: (N, H, D), out: (M, H, D). For head h the rows of Q_h are
// H*D elements apart and start at offset h*D, so the transposes of the
// reference are free: the fused flash-attention kernel below is simply
// launched with those strides (grid.y = heads). No mask; M may differ from N.
#include <cuda_runtime.h>
#include <cfloat>

// ---------------------------------------------------------------------------
// Generic FlashAttention-style forward for one (query-row block, head):
//   out = softmax(q k^T * scale) v, head_dim <= 1024, arbitrary row strides.
// - 4 warps = 4 query rows per block; lane l scores key l of each 32-key tile;
// - K and V tiles are streamed through shared memory in 128-wide slices of the
//   head dimension, so shared memory does not grow with head_dim;
// - online softmax (running max / sum, accumulator rescaling per tile);
// - lane l accumulates output columns l, l + 32, ... in registers (the slice
//   loops are unrolled so the accumulator indices are compile-time constants).
// ---------------------------------------------------------------------------
constexpr int kFlashWarps = 4;
constexpr int kFlashTile = 32;
constexpr int kDimSlice = 128;
constexpr int kMaxSlices = 8;  // head_dim <= 1024
constexpr int kColsPerSlice = kDimSlice / 32;

struct AttnGeom {
    int q_rows, kv_rows, head_dim;
    size_t q_stride, kv_stride, o_stride;  // elements between consecutive rows
    size_t q_head, kv_head, o_head;        // elements between consecutive heads
    float scale;
};

__global__ void __launch_bounds__(kFlashWarps * 32)
flashForward(const float* q, const float* k, const float* v, float* out, AttnGeom g) {
    extern __shared__ float smem[];
    float* q_s = smem;                                  // [warps][head_dim]
    float* k_s = q_s + kFlashWarps * g.head_dim;        // [32][kDimSlice + 1]
    float* v_s = k_s + kFlashTile * (kDimSlice + 1);    // [32][kDimSlice]
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int head = blockIdx.y;
    const int row = blockIdx.x * kFlashWarps + warp;
    const bool active = row < g.q_rows;
    const int hd = g.head_dim;
    const float* qh = q + head * g.q_head;
    const float* kh = k + head * g.kv_head;
    const float* vh = v + head * g.kv_head;

    for (int c = lane; c < hd; c += 32) q_s[warp * hd + c] = active ? qh[row * g.q_stride + c] * g.scale : 0.0f;

    float acc[kMaxSlices * kColsPerSlice];
#pragma unroll
    for (int r = 0; r < kMaxSlices * kColsPerSlice; ++r) acc[r] = 0.0f;
    float running_max = -FLT_MAX, running_sum = 0.0f;

    for (int key0 = 0; key0 < g.kv_rows; key0 += kFlashTile) {
        const int tile_keys = min(kFlashTile, g.kv_rows - key0);
        float score = 0.0f;
        for (int c0 = 0; c0 < hd; c0 += kDimSlice) {
            const int width = min(kDimSlice, hd - c0);
            __syncthreads();
            for (int i = threadIdx.x; i < kFlashTile * width; i += blockDim.x) {
                const int j = i / width, c = i % width;
                k_s[j * (kDimSlice + 1) + c] = j < tile_keys ? kh[(key0 + j) * g.kv_stride + c0 + c] : 0.0f;
            }
            __syncthreads();
            for (int c = 0; c < width; ++c) score = fmaf(q_s[warp * hd + c0 + c], k_s[lane * (kDimSlice + 1) + c], score);
        }
        if (lane >= tile_keys) score = -FLT_MAX;
        float tile_max = score;
        for (int o = 16; o > 0; o >>= 1) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, o));
        const float new_max = fmaxf(running_max, tile_max);
        const float corr = expf(running_max - new_max);
        const float p = lane < tile_keys ? expf(score - new_max) : 0.0f;
        float tile_sum = p;
        for (int o = 16; o > 0; o >>= 1) tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, o);
        running_sum = running_sum * corr + tile_sum;
        running_max = new_max;
#pragma unroll
        for (int r = 0; r < kMaxSlices * kColsPerSlice; ++r) acc[r] *= corr;

#pragma unroll
        for (int s = 0; s < kMaxSlices; ++s) {
            const int c0 = s * kDimSlice;
            if (c0 >= hd) break;
            const int width = min(kDimSlice, hd - c0);
            __syncthreads();
            for (int i = threadIdx.x; i < kFlashTile * width; i += blockDim.x) {
                const int j = i / width, c = i % width;
                v_s[j * kDimSlice + c] = j < tile_keys ? vh[(key0 + j) * g.kv_stride + c0 + c] : 0.0f;
            }
            __syncthreads();
            for (int j = 0; j < tile_keys; ++j) {
                const float pj = __shfl_sync(0xffffffffu, p, j);
#pragma unroll
                for (int rr = 0; rr < kColsPerSlice; ++rr) {
                    const int c = lane + 32 * rr;
                    if (c < width) acc[s * kColsPerSlice + rr] = fmaf(pj, v_s[j * kDimSlice + c], acc[s * kColsPerSlice + rr]);
                }
            }
        }
    }
    if (active) {
        const float inv = 1.0f / running_sum;
        float* oh = out + head * g.o_head + row * g.o_stride;
#pragma unroll
        for (int s = 0; s < kMaxSlices; ++s) {
#pragma unroll
            for (int rr = 0; rr < kColsPerSlice; ++rr) {
                const int c = s * kDimSlice + lane + 32 * rr;
                if (c < hd && lane + 32 * rr < kDimSlice) oh[c] = acc[s * kColsPerSlice + rr] * inv;
            }
        }
    }
}

static void launchFlash(const float* q, const float* k, const float* v, float* out, const AttnGeom& g, int heads) {
    const size_t smem = (static_cast<size_t>(kFlashWarps) * g.head_dim + kFlashTile * (kDimSlice + 1) +
                         kFlashTile * kDimSlice) * sizeof(float);
    cudaFuncSetAttribute(flashForward, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    const dim3 grid((g.q_rows + kFlashWarps - 1) / kFlashWarps, heads);
    flashForward<<<grid, kFlashWarps * 32, smem>>>(q, k, v, out, g);
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int H, int D) {
    const size_t row = static_cast<size_t>(H) * D;
    AttnGeom g{M, N, D, row, row, row, static_cast<size_t>(D), static_cast<size_t>(D), static_cast<size_t>(D),
               1.0f / sqrtf(static_cast<float>(D))};
    launchFlash(Q, K, V, output, g, H);
    cudaDeviceSynchronize();
}

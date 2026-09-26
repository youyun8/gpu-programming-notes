// Multi-Head Attention (LeetGPU)
// https://leetgpu.com/challenges/multi-head-attention
//
// Head i uses columns [i d_k, (i+1) d_k) of Q, K, V (row stride d_model) and
// writes the same columns of the output, so "split heads" and "concat" are
// just strides - no data movement. Each (row block, head) runs the fused
// flash-attention kernel below; d_k can be up to 1024 (h = 1), which is why the
// K/V tiles are streamed in 128-wide slices of the head dimension.
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

// Problem geometry: row counts, strides and per-head offsets, plus the softmax scale.
struct AttnGeom {
    int q_rows, kv_rows, head_dim;
    size_t q_stride, kv_stride, o_stride;  // elements between consecutive rows
    size_t q_head, kv_head, o_head;        // elements between consecutive heads
    float scale;
};

__global__ void __launch_bounds__(kFlashWarps * 32)
flashForward(const float* q, const float* k, const float* v, float* out, AttnGeom g) {
    // Dynamic shared memory: this block's 4 (pre-scaled) query rows, one K slice
    // (padded rows: conflict-free column reads) and one V slice.
    extern __shared__ float smem[];
    float* q_s = smem;                                  // [warps][head_dim]
    float* k_s = q_s + kFlashWarps * g.head_dim;        // [32][kDimSlice + 1]
    float* v_s = k_s + kFlashTile * (kDimSlice + 1);    // [32][kDimSlice]
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    // blockIdx.y selects the (batch, head) pair, blockIdx.x a group of 4 query rows (one per warp).
    const int head = blockIdx.y;
    const int row = blockIdx.x * kFlashWarps + warp;
    const bool active = row < g.q_rows;
    const int hd = g.head_dim;
    const float* qh = q + head * g.q_head;
    const float* kh = k + head * g.kv_head;
    const float* vh = v + head * g.kv_head;

    // Stage this warp's query row, multiplied by the scale 1/sqrt(E) once.
    for (int c = lane; c < hd; c += 32) q_s[warp * hd + c] = active ? qh[row * g.q_stride + c] * g.scale : 0.0f;

    // Output accumulator: lane l owns columns l, l + 32, ... (compile-time indices after unrolling),
    // plus the online-softmax state (running max and running sum).
    float acc[kMaxSlices * kColsPerSlice];
#pragma unroll
    for (int r = 0; r < kMaxSlices * kColsPerSlice; ++r) acc[r] = 0.0f;
    float running_max = -FLT_MAX, running_sum = 0.0f;

    // Stream the keys and values in tiles of 32.
    for (int key0 = 0; key0 < g.kv_rows; key0 += kFlashTile) {
        const int tile_keys = min(kFlashTile, g.kv_rows - key0);
        // Scores: lane l computes q . k_l for key l of the tile, over 128-wide slices of the head dim.
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
        // Online softmax: new max over the tile, rescale factor for the old state,
        // p = exp(score - new max), and the tile's sum of p.
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
        // Rescale the accumulator to the new max.
#pragma unroll
        for (int r = 0; r < kMaxSlices * kColsPerSlice; ++r) acc[r] *= corr;

        // acc += P V for this tile, one 128-wide slice of V at a time through shared memory.
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
            // Probability of key j is broadcast from lane j with a shuffle.
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
    // Normalize by the running sum and store the output row.
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

// Shared-memory size depends on the head dim; opt in above the 48 KB default when needed.
static void launchFlash(const float* q, const float* k, const float* v, float* out, const AttnGeom& g, int heads) {
    const size_t smem = (static_cast<size_t>(kFlashWarps) * g.head_dim + kFlashTile * (kDimSlice + 1) +
                         kFlashTile * kDimSlice) * sizeof(float);
    cudaFuncSetAttribute(flashForward, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    const dim3 grid((g.q_rows + kFlashWarps - 1) / kFlashWarps, heads);
    flashForward<<<grid, kFlashWarps * 32, smem>>>(q, k, v, out, g);
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h) {
    // Heads are interleaved in each row: head h of token i starts at i * d_model + h * dk,
    // so rows are d_model apart and heads dk apart. One flash launch covers all heads.
    const int dk = d_model / h;
    AttnGeom g{N, N, dk, static_cast<size_t>(d_model), static_cast<size_t>(d_model), static_cast<size_t>(d_model),
               static_cast<size_t>(dk), static_cast<size_t>(dk), static_cast<size_t>(dk), 1.0f / sqrtf(static_cast<float>(dk))};
    launchFlash(Q, K, V, output, g, h);
    cudaDeviceSynchronize();
}

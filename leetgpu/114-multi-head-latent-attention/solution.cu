// Multi-Head Latent Attention (decode, weight-absorbed) (LeetGPU)
// https://leetgpu.com/challenges/multi-head-latent-attention-decode
//
// DeepSeek-style MLA decode for one token:
//   q_lat[h] = q_nope[h] W_UK[h]                          (absorb W_UK into the query)
//   p[h, s]  = softmax_s((q_lat[h] . c_kv[s] + q_pe[h] . k_pe[s]) * scale)
//   out[h]   = (sum_s p[h, s] c_kv[s]) W_UV[h]             (attend in latent space)
// The cache row [c_kv | k_pe] is exactly the "key" for the concatenated query
// [q_lat | q_pe], and its c_kv prefix is the "value". So after a small
// absorption GEMV, the core is ordinary attention with head_dim R + rope
// (<= 576) for scores and R (<= 512) for values. All heads share the same
// cache (MQA-like), which makes K/V tiles highly reusable across the 4 heads
// handled by each block.
//   1. absorbQuery: one thread per (h, r) dot over head_dim, plus copy of q_pe;
//   2. latentAttention: flash-style, 4 heads per block, cache rows streamed
//      through shared memory in 32-row tiles and 128-wide column slices;
//   3. upProject: one thread per (h, j), dot over R with W_UV[h].
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kWarps = 4;
constexpr int kTile = 32;
constexpr int kSlice = 128;
constexpr int kMaxSlices = 4;  // value dim R <= 512
constexpr int kColsPerSlice = kSlice / 32;

__global__ void absorbQuery(const float* q, const float* w_uk, float* q_cat, int heads, int hd, int rank, int rope) {
    const int qdim = rank + rope;
    const int total = heads * qdim;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += gridDim.x * blockDim.x) {
        const int h = t / qdim, r = t % qdim;
        const float* qh = q + static_cast<size_t>(h) * (hd + rope);
        float v;
        if (r < rank) {
            v = 0.0f;
            const float* w = w_uk + static_cast<size_t>(h) * hd * rank + r;
            for (int i = 0; i < hd; ++i) v = fmaf(qh[i], w[static_cast<size_t>(i) * rank], v);
        } else {
            v = qh[hd + (r - rank)];
        }
        q_cat[t] = v;
    }
}

__global__ void __launch_bounds__(kWarps * 32)
latentAttention(const float* q_cat, const float* cache, float* latent_out, int heads, int seq, int rank, int rope, float scale) {
    extern __shared__ float smem[];
    const int qdim = rank + rope;
    float* q_s = smem;                               // [warps][qdim]
    float* t_s = q_s + kWarps * qdim;                // [32][kSlice + 1] (K slice, then V slice)
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int head = blockIdx.x * kWarps + warp;
    const bool active = head < heads;
    for (int c = lane; c < qdim; c += 32) q_s[warp * qdim + c] = active ? q_cat[static_cast<size_t>(head) * qdim + c] * scale : 0.0f;

    float acc[kMaxSlices * kColsPerSlice];
#pragma unroll
    for (int r = 0; r < kMaxSlices * kColsPerSlice; ++r) acc[r] = 0.0f;
    float mx = -FLT_MAX, sum = 0.0f;
    for (int j0 = 0; j0 < seq; j0 += kTile) {
        const int keys = min(kTile, seq - j0);
        float score = 0.0f;
        for (int c0 = 0; c0 < qdim; c0 += kSlice) {
            const int width = min(kSlice, qdim - c0);
            __syncthreads();
            for (int i = threadIdx.x; i < kTile * width; i += blockDim.x) {
                const int j = i / width, c = i % width;
                t_s[j * (kSlice + 1) + c] = j < keys ? cache[static_cast<size_t>(j0 + j) * qdim + c0 + c] : 0.0f;
            }
            __syncthreads();
            for (int c = 0; c < width; ++c) score = fmaf(q_s[warp * qdim + c0 + c], t_s[lane * (kSlice + 1) + c], score);
        }
        if (lane >= keys) score = -FLT_MAX;
        float tile_max = score;
        for (int o = 16; o > 0; o >>= 1) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, o));
        const float new_mx = fmaxf(mx, tile_max);
        const float corr = expf(mx - new_mx);
        const float p = lane < keys ? expf(score - new_mx) : 0.0f;
        float tile_sum = p;
        for (int o = 16; o > 0; o >>= 1) tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, o);
        sum = sum * corr + tile_sum;
        mx = new_mx;
#pragma unroll
        for (int r = 0; r < kMaxSlices * kColsPerSlice; ++r) acc[r] *= corr;
#pragma unroll
        for (int s = 0; s < kMaxSlices; ++s) {
            const int c0 = s * kSlice;
            if (c0 >= rank) break;
            const int width = min(kSlice, rank - c0);
            __syncthreads();
            for (int i = threadIdx.x; i < kTile * width; i += blockDim.x) {
                const int j = i / width, c = i % width;
                t_s[j * (kSlice + 1) + c] = j < keys ? cache[static_cast<size_t>(j0 + j) * qdim + c0 + c] : 0.0f;
            }
            __syncthreads();
            for (int j = 0; j < keys; ++j) {
                const float pj = __shfl_sync(0xffffffffu, p, j);
#pragma unroll
                for (int rr = 0; rr < kColsPerSlice; ++rr) {
                    const int c = lane + 32 * rr;
                    if (c < width) acc[s * kColsPerSlice + rr] = fmaf(pj, t_s[j * (kSlice + 1) + c], acc[s * kColsPerSlice + rr]);
                }
            }
        }
    }
    if (active) {
        const float inv = 1.0f / sum;
#pragma unroll
        for (int s = 0; s < kMaxSlices; ++s)
#pragma unroll
            for (int rr = 0; rr < kColsPerSlice; ++rr) {
                const int c = s * kSlice + lane + 32 * rr;
                if (c < rank) latent_out[static_cast<size_t>(head) * rank + c] = acc[s * kColsPerSlice + rr] * inv;
            }
    }
}

__global__ void upProject(const float* latent, const float* w_uv, float* out, int heads, int rank, int hd) {
    const int total = heads * hd;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += gridDim.x * blockDim.x) {
        const int h = t / hd, j = t % hd;
        const float* l = latent + static_cast<size_t>(h) * rank;
        const float* w = w_uv + static_cast<size_t>(h) * rank * hd + j;
        float v = 0.0f;
        for (int r = 0; r < rank; ++r) v = fmaf(l[r], w[static_cast<size_t>(r) * hd], v);
        out[t] = v;
    }
}

// q, kv_cache, W_UK, W_UV, output are device pointers
extern "C" void solve(const float* q, const float* kv_cache, const float* W_UK, const float* W_UV, float* output,
                      int num_heads, int seq_len, int kv_lora_rank, int head_dim, int rope_dim) {
    const int qdim = kv_lora_rank + rope_dim;
    float* buf = nullptr;
    cudaMalloc(&buf, static_cast<size_t>(num_heads) * (qdim + kv_lora_rank) * sizeof(float));
    float* q_cat = buf;
    float* latent = buf + static_cast<size_t>(num_heads) * qdim;
    absorbQuery<<<(num_heads * qdim + 255) / 256, 256>>>(q, W_UK, q_cat, num_heads, head_dim, kv_lora_rank, rope_dim);
    const size_t smem = (static_cast<size_t>(kWarps) * qdim + kTile * (kSlice + 1)) * sizeof(float);
    cudaFuncSetAttribute(latentAttention, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    latentAttention<<<(num_heads + kWarps - 1) / kWarps, kWarps * 32, smem>>>(
        q_cat, kv_cache, latent, num_heads, seq_len, kv_lora_rank, rope_dim, 1.0f / sqrtf(static_cast<float>(head_dim + rope_dim)));
    upProject<<<(num_heads * head_dim + 255) / 256, 256>>>(latent, W_UV, output, num_heads, kv_lora_rank, head_dim);
    cudaDeviceSynchronize();
    cudaFree(buf);
}

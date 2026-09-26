// INT8 KV-Cache Attention (decode) (LeetGPU)
// https://leetgpu.com/challenges/int8-kv-cache-attention
//
// One query token per head against a long int8 KV cache (per-token scales).
// Decode attention is bandwidth-bound and has little parallelism per head, so
// this uses flash-decoding (split-K over the sequence):
//   1. partialAttention: grid (chunks, heads); each block handles 256 keys:
//      scores (one warp per key, lanes over head_dim, dequantized on the fly),
//      then a local softmax (max m, sum l) and acc[d] = sum p_j V_j[d]
//      (threads over d, coalesced int8 reads);
//   2. combine: per head, merge the chunk partials with the usual rescaling.
#include <cstdint>
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kChunk = 256;
constexpr int kThreads = 256;

__global__ void partialAttention(const float* q, const int8_t* k, const int8_t* v, const float* k_scale,
                                 const float* v_scale, float* part_m, float* part_l, float* part_acc, int seq, int d,
                                 float scale) {
    // Split-KV decode attention (flash-decoding): each block handles one head and one
    // chunk of 256 cached keys; a second kernel merges the chunks.
    __shared__ float s_p[kChunk];
    __shared__ float s_red[kThreads / 32];
    const int head = blockIdx.y;
    const int chunk = blockIdx.x;
    const int key0 = chunk * kChunk;
    const int keys = min(kChunk, seq - key0);
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float* qh = q + static_cast<size_t>(head) * d;

    // Scores: one warp per key, int8 K dequantized on the fly (per-token scale), warp dot product.
    for (int j = warp; j < keys; j += kThreads / 32) {
        const size_t tok = static_cast<size_t>(head) * seq + key0 + j;
        const int8_t* kr = k + tok * d;
        float s = 0.0f;
        for (int c = lane; c < d; c += 32) s = fmaf(qh[c], static_cast<float>(kr[c]), s);
        for (int offset = 16; offset > 0; offset >>= 1) s += __shfl_xor_sync(0xffffffffu, s, offset);
        if (lane == 0) s_p[j] = s * k_scale[tok] * scale;
    }
    __syncthreads();

    // Chunk maximum (block reduction).
    float m = -FLT_MAX;
    for (int j = threadIdx.x; j < keys; j += kThreads) m = fmaxf(m, s_p[j]);
    for (int offset = 16; offset > 0; offset >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, offset));
    if (lane == 0) s_red[warp] = m;
    __syncthreads();
    m = s_red[0];
    for (int w = 1; w < kThreads / 32; ++w) m = fmaxf(m, s_red[w]);
    __syncthreads();

    // p_j = exp(s_j - m) and the chunk's sum l.
    float l = 0.0f;
    for (int j = threadIdx.x; j < keys; j += kThreads) {
        const float p = expf(s_p[j] - m);
        s_p[j] = p;
        l += p;
    }
    for (int offset = 16; offset > 0; offset >>= 1) l += __shfl_xor_sync(0xffffffffu, l, offset);
    if (lane == 0) s_red[warp] = l;
    __syncthreads();
    l = 0.0f;
    for (int w = 0; w < kThreads / 32; ++w) l += s_red[w];

    // Unnormalized partial output sum_j p_j v_j (int8 V dequantized with its per-token scale),
    // plus the chunk's (m, l) for the merge.
    const size_t part = static_cast<size_t>(head) * gridDim.x + chunk;
    for (int c = threadIdx.x; c < d; c += kThreads) {
        float acc = 0.0f;
        for (int j = 0; j < keys; ++j) {
            const size_t tok = static_cast<size_t>(head) * seq + key0 + j;
            acc = fmaf(s_p[j] * v_scale[tok], static_cast<float>(v[tok * d + c]), acc);
        }
        part_acc[part * d + c] = acc;
    }
    if (threadIdx.x == 0) {
        part_m[part] = m;
        part_l[part] = l;
    }
}

__global__ void combine(const float* part_m, const float* part_l, const float* part_acc, float* out, int chunks, int d) {
    // Merge (one block per head): rescale every chunk to the global max, sum, and normalize.
    const int head = blockIdx.x;
    float m = -FLT_MAX;
    for (int i = 0; i < chunks; ++i) m = fmaxf(m, part_m[static_cast<size_t>(head) * chunks + i]);
    float l = 0.0f;
    for (int i = 0; i < chunks; ++i) {
        const size_t p = static_cast<size_t>(head) * chunks + i;
        l += part_l[p] * expf(part_m[p] - m);
    }
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < chunks; ++i) {
            const size_t p = static_cast<size_t>(head) * chunks + i;
            acc += part_acc[p * d + c] * expf(part_m[p] - m);
        }
        out[static_cast<size_t>(head) * d + c] = acc / l;
    }
}

// Q, K_int8, V_int8, k_scale, v_scale, output are device pointers
extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8, const float* k_scale,
                      const float* v_scale, float* output, int num_heads, int seq_len, int head_dim) {
    // Scratch for the per-chunk (m, l, acc), then the two kernels.
    const int chunks = (seq_len + kChunk - 1) / kChunk;
    const size_t parts = static_cast<size_t>(num_heads) * chunks;
    float* buf = nullptr;
    cudaMalloc(&buf, parts * (2 + head_dim) * sizeof(float));
    float* part_m = buf;
    float* part_l = buf + parts;
    float* part_acc = buf + 2 * parts;
    partialAttention<<<dim3(chunks, num_heads), kThreads>>>(Q, K_int8, V_int8, k_scale, v_scale, part_m, part_l, part_acc,
                                                           seq_len, head_dim, 1.0f / sqrtf(static_cast<float>(head_dim)));
    combine<<<num_heads, 256>>>(part_m, part_l, part_acc, output, chunks, head_dim);
    cudaDeviceSynchronize();
    cudaFree(buf);
}

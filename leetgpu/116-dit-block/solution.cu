// DiT Block (LeetGPU)
// https://leetgpu.com/challenges/diffusion-transformer-block
//
// Diffusion-Transformer block with adaLN-Zero conditioning (D = 512, 8 heads
// of 64, MLP = 2048; weights stored (out, in) -> "NT" GEMMs):
//   [shift1, scale1, gate1, shift2, scale2, gate2] = silu(c) W_ada^T + b_ada
//   x1  = x  + gate1 * (Attn(LN(x) (1 + scale1) + shift1) W_o^T + b_o)
//   out = x1 + gate2 * (gelu(LN(x1) (1 + scale2) + shift2) W1^T + b1) W2^T + b2)
// The per-sample modulation is fused into the LayerNorm kernel, and the
// gated residuals into the GEMM epilogues.
#include <cuda_runtime.h>
#include <cfloat>

// ---------------------------------------------------------------------------
// Building blocks shared by the transformer-block kernels in this file.
// ---------------------------------------------------------------------------
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kGemmThreads = 256;

// C[r][c] = epi(sum_k A[r][k] * B'[k][c], r, c)
//   A: rows x inner, row stride lda.
//   B' = B^T (kTransB, B stored cols x inner, "NT") or B (inner x cols, "NN"), row stride ldb.
// 64 x 64 block tile, 16-wide K slices in shared memory, 4 x 4 outputs per thread.
template <bool kTransB, class Epi>
__global__ void __launch_bounds__(kGemmThreads)
gemmKernel(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int rows, int inner, int cols, Epi epi) {
    // Shared-memory staging buffers for one K-slice. A is stored transposed
    // ([k][m]) so that both operands are read along rows; +4 padding avoids bank conflicts.
    __shared__ float a_tile[kTileK][kTileM + 4];
    __shared__ float b_tile[kTileK][kTileN + 4];
    // Thread coordinates: a 16 x 16 grid of threads, each owning a 4 x 4 patch of
    // C at rows ty + 16i and columns tx + 16j (stride 16 keeps stores coalesced).
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    // Top-left corner of this block's 64 x 64 output tile.
    const int row0 = blockIdx.y * kTileM, col0 = blockIdx.x * kTileN;
    // Per-thread accumulators, kept in registers (fully unrolled loops below).
    float acc[4][4] = {};
    // Main loop over the reduction dimension, 16 columns of A / rows of B at a time.
    for (int k0 = 0; k0 < inner; k0 += kTileK) {
        // Stage the 64 x 16 panel of A (zero-filled outside the matrix), transposing it.
        for (int i = tid; i < kTileM * kTileK; i += kGemmThreads) {
            const int r = i / kTileK, kk = i % kTileK;
            a_tile[kk][r] = (row0 + r < rows && k0 + kk < inner) ? a[static_cast<size_t>(row0 + r) * lda + k0 + kk] : 0.0f;
        }
        // Stage the 16 x 64 panel of B. With kTransB, B is stored (cols x inner) and is
        // transposed while loading; otherwise it is copied row by row (coalesced).
        for (int i = tid; i < kTileK * kTileN; i += kGemmThreads) {
            if (kTransB) {
                const int cc = i / kTileK, kk = i % kTileK;
                b_tile[kk][cc] = (col0 + cc < cols && k0 + kk < inner) ? b[static_cast<size_t>(col0 + cc) * ldb + k0 + kk] : 0.0f;
            } else {
                const int kk = i / kTileN, cc = i % kTileN;
                b_tile[kk][cc] = (k0 + kk < inner && col0 + cc < cols) ? b[static_cast<size_t>(k0 + kk) * ldb + col0 + cc] : 0.0f;
            }
        }
        // Both panels must be complete before any thread reads them.
        __syncthreads();
        // Inner product: per k, load 4 values of A and 4 of B into registers and do a
        // 4 x 4 outer product (16 FMAs for 8 shared loads).
#pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            float af[4], bf[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) af[i] = a_tile[kk][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) bf[j] = b_tile[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(af[i], bf[j], acc[i][j]);
        }
        // Wait until every thread is done with the panels before the next slice overwrites them.
        __syncthreads();
    }
    // Epilogue: apply the fused operation (bias, activation, ...) and store in bounds.
    for (int i = 0; i < 4; ++i) {
        const int r = row0 + ty + 16 * i;
        if (r >= rows) continue;
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + tx + 16 * j;
            if (col < cols) c[static_cast<size_t>(r) * ldc + col] = epi(acc[i][j], r, col);
        }
    }
}

// Host helper: one block per 64 x 64 output tile.
template <bool kTransB, class Epi>
static void gemm(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int rows, int inner, int cols, Epi epi) {
    const dim3 grid((cols + kTileN - 1) / kTileN, (rows + kTileM - 1) / kTileM);
    gemmKernel<kTransB, Epi><<<grid, kGemmThreads>>>(a, lda, b, ldb, c, ldc, rows, inner, cols, epi);
}

struct NoEpi {
    __device__ float operator()(float v, int, int) const { return v; }
};
struct BiasEpi {
    const float* bias;
    __device__ float operator()(float v, int, int c) const { return v + bias[c]; }
};

// GELU, tanh approximation.
__device__ __forceinline__ float geluTanh(float x) {
    const float k0 = 0.7978845608028654f;  // sqrt(2 / pi)
    return 0.5f * x * (1.0f + tanhf(k0 * (x + 0.044715f * x * x * x)));
}

// Flash attention over rows of strided Q/K/V (head_dim <= 128).
// grid = (row blocks, heads, batch); query head h reads KV head h / group.
struct AttnParams {
    int q_rows, kv_rows, head_dim, group, causal;
    size_t q_stride, k_stride, v_stride, o_stride;  // row strides
    size_t q_head, k_head, v_head, o_head;          // head offsets
    size_t q_batch, k_batch, v_batch, o_batch;      // batch offsets
    float scale;
};

constexpr int kAttnWarps = 4;
constexpr int kAttnTile = 32;
constexpr int kAttnMaxDim = 128;
constexpr int kAttnPitch = kAttnMaxDim + 1;
constexpr int kAttnCols = kAttnMaxDim / 32;

__global__ void __launch_bounds__(kAttnWarps * 32) attentionKernel(const float* q, const float* k, const float* v, float* out, AttnParams p) {
    // Generic flash-style attention: shared tiles of 32 keys and values (odd pitch) and the
    // block's query rows. Strides and head/batch offsets come from AttnParams, so the same
    // kernel serves self/cross attention, GQA (kv_head = head / group) and causal masking.
    __shared__ float k_t[kAttnTile][kAttnPitch];
    __shared__ float v_t[kAttnTile][kAttnPitch];
    __shared__ float q_s[kAttnWarps][kAttnMaxDim];
    // One warp per query row; grid = (row groups, heads, batch).
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int head = blockIdx.y, batch = blockIdx.z, kv_head = head / p.group;
    const int row = blockIdx.x * kAttnWarps + warp;
    const bool active = row < p.q_rows;
    const int d = p.head_dim;
    const float* qb = q + batch * p.q_batch + head * p.q_head;
    const float* kb = k + batch * p.k_batch + kv_head * p.k_head;
    const float* vb = v + batch * p.v_batch + kv_head * p.v_head;
    // Stage the query row, pre-scaled; with a causal mask no key after the block's last row is needed.
    for (int c = lane; c < d; c += 32) q_s[warp][c] = active ? qb[row * p.q_stride + c] * p.scale : 0.0f;
    const int last_row = min(p.q_rows - 1, static_cast<int>(blockIdx.x) * kAttnWarps + kAttnWarps - 1);
    const int key_end = p.causal ? min(p.kv_rows, last_row + 1) : p.kv_rows;
    float acc[kAttnCols] = {};
    float mx = -FLT_MAX, sum = 0.0f;
    // Stream K and V in tiles of 32 keys (zero rows past the end).
    for (int j0 = 0; j0 < key_end; j0 += kAttnTile) {
        __syncthreads();
        for (int i = threadIdx.x; i < kAttnTile * d; i += blockDim.x) {
            const int r = i / d, c = i % d;
            const bool ok = j0 + r < key_end;
            k_t[r][c] = ok ? kb[(j0 + r) * p.k_stride + c] : 0.0f;
            v_t[r][c] = ok ? vb[(j0 + r) * p.v_stride + c] : 0.0f;
        }
        __syncthreads();
        // Lane l scores key j0 + l (masked if causal and j > row).
        const int j = j0 + lane;
        const bool allowed = active && j < key_end && (!p.causal || j <= row);
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
        const float pr = allowed ? expf(s - new_mx) : 0.0f;
        float tile_sum = pr;
        for (int o = 16; o > 0; o >>= 1) tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, o);
        sum = sum * corr + tile_sum;
        mx = new_mx;
        // Rescale the accumulator, then add P V (probability of key jj broadcast from lane jj).
        for (int r = 0; r < kAttnCols; ++r) acc[r] *= corr;
        for (int jj = 0; jj < min(kAttnTile, key_end - j0); ++jj) {
            const float pj = __shfl_sync(0xffffffffu, pr, jj);
            for (int r = 0; r < kAttnCols; ++r) {
                const int c = lane + 32 * r;
                if (c < d) acc[r] = fmaf(pj, v_t[jj][c], acc[r]);
            }
        }
    }
    // Normalize and store the output row.
    if (active) {
        float* ob = out + batch * p.o_batch + head * p.o_head + row * p.o_stride;
        const float inv = 1.0f / sum;
        for (int r = 0; r < kAttnCols; ++r) {
            const int c = lane + 32 * r;
            if (c < d) ob[c] = acc[r] * inv;
        }
    }
}

// One warp per query row, 4 or 8 rows per block, one grid slice per head and batch.
static void attention(const float* q, const float* k, const float* v, float* out, const AttnParams& p, int heads, int batch) {
    const dim3 grid((p.q_rows + kAttnWarps - 1) / kAttnWarps, heads, batch);
    attentionKernel<<<grid, kAttnWarps * 32>>>(q, k, v, out, p);
}

// Warp-per-row LayerNorm: y = (x - mean) * rstd * w + b  (w, b optional).
__global__ void layerNormRows(const float* x, float* y, const float* w, const float* b, int rows, int d, float eps) {
    // LayerNorm, one warp per row: mean, centered variance, then the optional affine
    // parameters (w or b may be null).
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const float* xr = x + static_cast<size_t>(row) * d;
    float s = 0.0f;
    for (int c = lane; c < d; c += 32) s += xr[c];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s / d;
    float sq = 0.0f;
    for (int c = lane; c < d; c += 32) {
        const float t = xr[c] - mean;
        sq += t * t;
    }
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    const float rstd = rsqrtf(sq / d + eps);
    float* yr = y + static_cast<size_t>(row) * d;
    for (int c = lane; c < d; c += 32) {
        float t = (xr[c] - mean) * rstd;
        if (w) t = t * w[c];
        if (b) t = t + b[c];
        yr[c] = t;
    }
}

// Warp-per-row RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w.
__global__ void rmsNormRows(const float* x, float* y, const float* w, int rows, int d, float eps) {
    // RMSNorm, one warp per row: y = x / rms(x) * w.
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const float* xr = x + static_cast<size_t>(row) * d;
    float sq = 0.0f;
    for (int c = lane; c < d; c += 32) sq += xr[c] * xr[c];
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    const float r = rsqrtf(sq / d + eps);
    float* yr = y + static_cast<size_t>(row) * d;
    for (int c = lane; c < d; c += 32) yr[c] = xr[c] * r * w[c];
}

static int rowBlocks(int rows) { return (rows + 7) / 8; }  // 8 warps (rows) per 256-thread block

constexpr int kD = 512;
constexpr int kHeads = 8;
constexpr int kDh = kD / kHeads;
constexpr int kMlp = 4 * kD;
constexpr float kEps = 1e-6f;
constexpr size_t kWada = 0;
constexpr size_t kBada = kWada + 6 * static_cast<size_t>(kD) * kD;
constexpr size_t kWqkv = kBada + 6 * kD;
constexpr size_t kBqkv = kWqkv + 3 * static_cast<size_t>(kD) * kD;
constexpr size_t kWo = kBqkv + 3 * kD;
constexpr size_t kBo = kWo + static_cast<size_t>(kD) * kD;
constexpr size_t kWfc1 = kBo + kD;
constexpr size_t kBfc1 = kWfc1 + static_cast<size_t>(kMlp) * kD;
constexpr size_t kWfc2 = kBfc1 + kMlp;
constexpr size_t kBfc2 = kWfc2 + static_cast<size_t>(kD) * kMlp;

struct BiasGeluEpi {
    const float* bias;
    __device__ float operator()(float v, int, int c) const { return geluTanh(v + bias[c]); }
};
// out = residual + gate[batch][c] * (v + bias[c]), batch = row / seq.
// adaLN-Zero residual: x + gate * (y + bias), with the gate taken from this sample's
// chunk `gate_chunk` of the 6-way modulation vector.
struct GatedResidualEpi {
    const float* bias;
    const float* residual;
    const float* mod;  // batch x 6D
    int gate_chunk;
    int seq;
    __device__ float operator()(float v, int r, int c) const {
        const float gate = mod[static_cast<size_t>(r / seq) * 6 * kD + gate_chunk * kD + c];
        return residual[static_cast<size_t>(r) * kD + c] + gate * (v + bias[c]);
    }
};

// SiLU of the conditioning vector before the modulation GEMM.
__global__ void siluVec(const float* in, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] / (1.0f + expf(-in[i]));
}

// LayerNorm (no affine) followed by h * (1 + scale[b]) + shift[b]; warp per row.
__global__ void modulatedLayerNorm(const float* x, float* y, const float* mod, int shift_chunk, int scale_chunk, int rows, int seq) {
    // Modulated LayerNorm (adaLN), one warp per row: normalize without affine parameters,
    // then x_hat * (1 + scale) + shift with this sample's scale and shift chunks.
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const float* xr = x + static_cast<size_t>(row) * kD;
    float s = 0.0f;
    for (int c = lane; c < kD; c += 32) s += xr[c];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s / kD;
    float sq = 0.0f;
    for (int c = lane; c < kD; c += 32) {
        const float t = xr[c] - mean;
        sq += t * t;
    }
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    const float rstd = rsqrtf(sq / kD + kEps);
    const float* mb = mod + static_cast<size_t>(row / seq) * 6 * kD;
    float* yr = y + static_cast<size_t>(row) * kD;
    for (int c = lane; c < kD; c += 32) yr[c] = (xr[c] - mean) * rstd * (1.0f + mb[scale_chunk * kD + c]) + mb[shift_chunk * kD + c];
}

// x, c, output, weights are device pointers
extern "C" void solve(const float* x, const float* c, float* output, const float* weights, int batch_size, int seq_len) {
    const int rows = batch_size * seq_len;
    const float* w = weights;
    float* buf = nullptr;
    // One scratch allocation: conditioning, modulation vectors and all activations.
    const size_t floats = static_cast<size_t>(batch_size) * kD * 7 +
                          static_cast<size_t>(rows) * (kD + 3 * kD + kD + kD + kMlp);
    cudaMalloc(&buf, floats * sizeof(float));
    float* c_silu = buf;                                        // B x D
    float* mod = c_silu + static_cast<size_t>(batch_size) * kD; // B x 6D
    float* h = mod + static_cast<size_t>(batch_size) * 6 * kD;  // rows x D
    float* qkv = h + static_cast<size_t>(rows) * kD;            // rows x 3D
    float* attn = qkv + static_cast<size_t>(rows) * 3 * kD;     // rows x D
    float* x1 = attn + static_cast<size_t>(rows) * kD;          // rows x D
    float* fc1 = x1 + static_cast<size_t>(rows) * kD;           // rows x MLP

    // adaLN modulation: mod = SiLU(c) W_ada + b, six D-sized chunks per sample
    // (shift/scale/gate for attention, then for the MLP).
    siluVec<<<(batch_size * kD + 255) / 256, 256>>>(c, c_silu, batch_size * kD);
    gemm<true>(c_silu, kD, w + kWada, kD, mod, 6 * kD, batch_size, kD, 6 * kD, BiasEpi{w + kBada});

    // Attention sub-block: modulated LN (chunks 0, 1) -> QKV projection.
    modulatedLayerNorm<<<rowBlocks(rows), 256>>>(x, h, mod, 0, 1, rows, seq_len);
    gemm<true>(h, kD, w + kWqkv, kD, qkv, 3 * kD, rows, kD, 3 * kD, BiasEpi{w + kBqkv});
    // Bidirectional multi-head attention per sample (batch offsets in AttnParams).
    AttnParams p{};
    p.q_rows = seq_len;
    p.kv_rows = seq_len;
    p.head_dim = kDh;
    p.group = 1;
    p.causal = 0;
    p.q_stride = p.k_stride = p.v_stride = 3 * kD;
    p.o_stride = kD;
    p.q_head = p.k_head = p.v_head = p.o_head = kDh;
    p.q_batch = p.k_batch = p.v_batch = static_cast<size_t>(seq_len) * 3 * kD;
    p.o_batch = static_cast<size_t>(seq_len) * kD;
    p.scale = 1.0f / sqrtf(static_cast<float>(kDh));
    attention(qkv, qkv + kD, qkv + 2 * kD, attn, p, kHeads, batch_size);
    // Output projection, gated residual with chunk 2.
    gemm<true>(attn, kD, w + kWo, kD, x1, kD, rows, kD, kD, GatedResidualEpi{w + kBo, x, mod, 2, seq_len});

    // MLP sub-block: modulated LN (chunks 3, 4) -> FC1 + GELU -> FC2, gated residual with chunk 5.
    modulatedLayerNorm<<<rowBlocks(rows), 256>>>(x1, h, mod, 3, 4, rows, seq_len);
    gemm<true>(h, kD, w + kWfc1, kD, fc1, kMlp, rows, kD, kMlp, BiasGeluEpi{w + kBfc1});
    gemm<true>(fc1, kMlp, w + kWfc2, kMlp, output, kD, rows, kMlp, kD, GatedResidualEpi{w + kBfc2, x1, mod, 5, seq_len});
    cudaDeviceSynchronize();
    cudaFree(buf);
}

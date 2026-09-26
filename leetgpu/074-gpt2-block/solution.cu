// GPT-2 Transformer Block (LeetGPU)
// https://leetgpu.com/challenges/gpt-2-transformer-block
//
// Pre-LN block, D = 768, 12 heads of 64, FFN = 3072, weights stored (in, out):
//   h   = x + (Attn(LN1(x)) W_o + b_o)
//   out = h + (gelu_tanh(LN2(h) W_fc + b_fc) W_proj + b_proj)
// Kernels: LayerNorm (warp per row) -> QKV GEMM (+bias) -> flash attention
// reading Q/K/V straight out of the packed qkv rows (strides, no reshapes) ->
// out-projection GEMM with bias + residual fused -> LayerNorm -> FC GEMM with
// bias + GELU fused -> projection GEMM with bias + residual fused.
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
    __shared__ float a_tile[kTileK][kTileM + 4];
    __shared__ float b_tile[kTileK][kTileN + 4];
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int row0 = blockIdx.y * kTileM, col0 = blockIdx.x * kTileN;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < inner; k0 += kTileK) {
        for (int i = tid; i < kTileM * kTileK; i += kGemmThreads) {
            const int r = i / kTileK, kk = i % kTileK;
            a_tile[kk][r] = (row0 + r < rows && k0 + kk < inner) ? a[static_cast<size_t>(row0 + r) * lda + k0 + kk] : 0.0f;
        }
        for (int i = tid; i < kTileK * kTileN; i += kGemmThreads) {
            if (kTransB) {
                const int cc = i / kTileK, kk = i % kTileK;
                b_tile[kk][cc] = (col0 + cc < cols && k0 + kk < inner) ? b[static_cast<size_t>(col0 + cc) * ldb + k0 + kk] : 0.0f;
            } else {
                const int kk = i / kTileN, cc = i % kTileN;
                b_tile[kk][cc] = (k0 + kk < inner && col0 + cc < cols) ? b[static_cast<size_t>(k0 + kk) * ldb + col0 + cc] : 0.0f;
            }
        }
        __syncthreads();
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
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i) {
        const int r = row0 + ty + 16 * i;
        if (r >= rows) continue;
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + tx + 16 * j;
            if (col < cols) c[static_cast<size_t>(r) * ldc + col] = epi(acc[i][j], r, col);
        }
    }
}

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
    __shared__ float k_t[kAttnTile][kAttnPitch];
    __shared__ float v_t[kAttnTile][kAttnPitch];
    __shared__ float q_s[kAttnWarps][kAttnMaxDim];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int head = blockIdx.y, batch = blockIdx.z, kv_head = head / p.group;
    const int row = blockIdx.x * kAttnWarps + warp;
    const bool active = row < p.q_rows;
    const int d = p.head_dim;
    const float* qb = q + batch * p.q_batch + head * p.q_head;
    const float* kb = k + batch * p.k_batch + kv_head * p.k_head;
    const float* vb = v + batch * p.v_batch + kv_head * p.v_head;
    for (int c = lane; c < d; c += 32) q_s[warp][c] = active ? qb[row * p.q_stride + c] * p.scale : 0.0f;
    const int last_row = min(p.q_rows - 1, static_cast<int>(blockIdx.x) * kAttnWarps + kAttnWarps - 1);
    const int key_end = p.causal ? min(p.kv_rows, last_row + 1) : p.kv_rows;
    float acc[kAttnCols] = {};
    float mx = -FLT_MAX, sum = 0.0f;
    for (int j0 = 0; j0 < key_end; j0 += kAttnTile) {
        __syncthreads();
        for (int i = threadIdx.x; i < kAttnTile * d; i += blockDim.x) {
            const int r = i / d, c = i % d;
            const bool ok = j0 + r < key_end;
            k_t[r][c] = ok ? kb[(j0 + r) * p.k_stride + c] : 0.0f;
            v_t[r][c] = ok ? vb[(j0 + r) * p.v_stride + c] : 0.0f;
        }
        __syncthreads();
        const int j = j0 + lane;
        const bool allowed = active && j < key_end && (!p.causal || j <= row);
        float s = -FLT_MAX;
        if (allowed) {
            s = 0.0f;
            for (int c = 0; c < d; ++c) s = fmaf(q_s[warp][c], k_t[lane][c], s);
        }
        float tile_max = s;
        for (int o = 16; o > 0; o >>= 1) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, o));
        const float new_mx = fmaxf(mx, tile_max);
        const float corr = expf(mx - new_mx);
        const float pr = allowed ? expf(s - new_mx) : 0.0f;
        float tile_sum = pr;
        for (int o = 16; o > 0; o >>= 1) tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, o);
        sum = sum * corr + tile_sum;
        mx = new_mx;
        for (int r = 0; r < kAttnCols; ++r) acc[r] *= corr;
        for (int jj = 0; jj < min(kAttnTile, key_end - j0); ++jj) {
            const float pj = __shfl_sync(0xffffffffu, pr, jj);
            for (int r = 0; r < kAttnCols; ++r) {
                const int c = lane + 32 * r;
                if (c < d) acc[r] = fmaf(pj, v_t[jj][c], acc[r]);
            }
        }
    }
    if (active) {
        float* ob = out + batch * p.o_batch + head * p.o_head + row * p.o_stride;
        const float inv = 1.0f / sum;
        for (int r = 0; r < kAttnCols; ++r) {
            const int c = lane + 32 * r;
            if (c < d) ob[c] = acc[r] * inv;
        }
    }
}

static void attention(const float* q, const float* k, const float* v, float* out, const AttnParams& p, int heads, int batch) {
    const dim3 grid((p.q_rows + kAttnWarps - 1) / kAttnWarps, heads, batch);
    attentionKernel<<<grid, kAttnWarps * 32>>>(q, k, v, out, p);
}

// Warp-per-row LayerNorm: y = (x - mean) * rstd * w + b  (w, b optional).
__global__ void layerNormRows(const float* x, float* y, const float* w, const float* b, int rows, int d, float eps) {
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

constexpr int kD = 768;
constexpr int kHeads = 12;
constexpr int kDh = kD / kHeads;
constexpr int kFfn = 3072;
constexpr size_t kLn1W = 0;
constexpr size_t kLn1B = kLn1W + kD;
constexpr size_t kWqkv = kLn1B + kD;
constexpr size_t kBqkv = kWqkv + static_cast<size_t>(kD) * 3 * kD;
constexpr size_t kWo = kBqkv + 3 * kD;
constexpr size_t kBo = kWo + static_cast<size_t>(kD) * kD;
constexpr size_t kLn2W = kBo + kD;
constexpr size_t kLn2B = kLn2W + kD;
constexpr size_t kWfc = kLn2B + kD;
constexpr size_t kBfc = kWfc + static_cast<size_t>(kD) * kFfn;
constexpr size_t kWproj = kBfc + kFfn;
constexpr size_t kBproj = kWproj + static_cast<size_t>(kFfn) * kD;

struct BiasResidualEpi {
    const float* bias;
    const float* residual;
    int ld;
    __device__ float operator()(float v, int r, int c) const { return residual[static_cast<size_t>(r) * ld + c] + v + bias[c]; }
};
struct BiasGeluEpi {
    const float* bias;
    __device__ float operator()(float v, int, int c) const { return geluTanh(v + bias[c]); }
};

// x, output, weights are device pointers
extern "C" void solve(const float* x, float* output, const float* weights, int seq_len) {
    const int s = seq_len;
    float* buf = nullptr;
    const size_t per_row = static_cast<size_t>(kD) * 3 + kD + kD + kD + kFfn;
    cudaMalloc(&buf, per_row * s * sizeof(float));
    float* qkv = buf;                                     // s x 3D
    float* xn = qkv + static_cast<size_t>(s) * 3 * kD;    // s x D (LN1, later LN2)
    float* attn = xn + static_cast<size_t>(s) * kD;       // s x D
    float* hidden = attn + static_cast<size_t>(s) * kD;   // s x D
    float* fc = hidden + static_cast<size_t>(s) * kD;     // s x FFN
    const float* w = weights;

    layerNormRows<<<rowBlocks(s), 256>>>(x, xn, w + kLn1W, w + kLn1B, s, kD, 1e-5f);
    gemm<false>(xn, kD, w + kWqkv, 3 * kD, qkv, 3 * kD, s, kD, 3 * kD, BiasEpi{w + kBqkv});
    AttnParams p{};
    p.q_rows = s;
    p.kv_rows = s;
    p.head_dim = kDh;
    p.group = 1;
    p.causal = 0;
    p.q_stride = p.k_stride = p.v_stride = 3 * kD;
    p.o_stride = kD;
    p.q_head = p.k_head = p.v_head = p.o_head = kDh;
    p.scale = 1.0f / sqrtf(static_cast<float>(kDh));
    attention(qkv, qkv + kD, qkv + 2 * kD, attn, p, kHeads, 1);
    gemm<false>(attn, kD, w + kWo, kD, hidden, kD, s, kD, kD, BiasResidualEpi{w + kBo, x, kD});
    layerNormRows<<<rowBlocks(s), 256>>>(hidden, xn, w + kLn2W, w + kLn2B, s, kD, 1e-5f);
    gemm<false>(xn, kD, w + kWfc, kFfn, fc, kFfn, s, kD, kFfn, BiasGeluEpi{w + kBfc});
    gemm<false>(fc, kFfn, w + kWproj, kD, output, kD, s, kFfn, kD, BiasResidualEpi{w + kBproj, hidden, kD});
    cudaDeviceSynchronize();
    cudaFree(buf);
}

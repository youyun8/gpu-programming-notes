// LLaMA Transformer Block (LeetGPU)
// https://leetgpu.com/challenges/llama-transformer-block
//
// D = 512, 8 query heads / 2 KV heads (GQA) of 64, SwiGLU FFN of 1408,
// RMSNorm, RoPE, causal attention; weights stored (out, in) -> "NT" GEMMs.
//   h   = x + Attn(RoPE(RMS1(x) Wqkv^T)) Wo^T
//   out = h + (silu(RMS2(h) Wg^T) * RMS2(h) Wu^T) Wd^T
// W_Q, W_K, W_V are contiguous in the weight buffer, so Q, K and V come out of
// ONE GEMM (768 output columns); likewise W_gate | W_up form one GEMM whose
// result is combined by a small SwiGLU kernel. Residuals are fused into the
// epilogues of the Wo and Wd GEMMs.
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

constexpr int kD = 512;
constexpr int kQHeads = 8;
constexpr int kKvHeads = 2;
constexpr int kDh = kD / kQHeads;
constexpr int kKvDim = kKvHeads * kDh;
constexpr int kQkvDim = kD + 2 * kKvDim;
constexpr int kFfn = 1408;
constexpr size_t kRms1 = 0;
constexpr size_t kWq = kRms1 + kD;
constexpr size_t kWo = kWq + static_cast<size_t>(kQkvDim) * kD;
constexpr size_t kRms2 = kWo + static_cast<size_t>(kD) * kD;
constexpr size_t kWgate = kRms2 + kD;
constexpr size_t kWdown = kWgate + 2 * static_cast<size_t>(kFfn) * kD;

struct ResidualEpi {
    const float* residual;
    int ld;
    __device__ float operator()(float v, int r, int c) const { return residual[static_cast<size_t>(r) * ld + c] + v; }
};

// Rotate-half RoPE in place on the Q heads and K heads of each qkv row.
__global__ void applyRope(float* qkv, const float* cos_t, const float* sin_t, int s) {
    const int half = kDh / 2;
    const int heads = kQHeads + kKvHeads;
    const int total = s * heads * half;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += gridDim.x * blockDim.x) {
        const int i = t % half;
        const int head = (t / half) % heads;
        const int row = t / (half * heads);
        float* base = qkv + static_cast<size_t>(row) * kQkvDim + head * kDh;  // K heads follow the Q heads
        const float c = cos_t[row * half + i], sn = sin_t[row * half + i];
        const float x1 = base[i], x2 = base[i + half];
        base[i] = x1 * c - x2 * sn;
        base[i + half] = x1 * sn + x2 * c;
    }
}

__global__ void swiglu(const float* gate_up, float* h, int s) {
    const int total = s * kFfn;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += gridDim.x * blockDim.x) {
        const int row = t / kFfn, j = t % kFfn;
        const float g = gate_up[static_cast<size_t>(row) * 2 * kFfn + j];
        const float u = gate_up[static_cast<size_t>(row) * 2 * kFfn + kFfn + j];
        h[t] = g / (1.0f + expf(-g)) * u;
    }
}

// x, output, weights, cos, sin are device pointers
extern "C" void solve(const float* x, float* output, const float* weights, const float* cos, const float* sin, int seq_len) {
    const int s = seq_len;
    float* buf = nullptr;
    const size_t per_row = static_cast<size_t>(kD) + kQkvDim + kD + kD + 2 * kFfn + kFfn;
    cudaMalloc(&buf, per_row * s * sizeof(float));
    float* xn = buf;                                        // s x D (RMS1, later RMS2)
    float* qkv = xn + static_cast<size_t>(s) * kD;          // s x 768
    float* attn = qkv + static_cast<size_t>(s) * kQkvDim;   // s x D
    float* hidden = attn + static_cast<size_t>(s) * kD;     // s x D
    float* gate_up = hidden + static_cast<size_t>(s) * kD;  // s x 2 FFN
    float* ffn = gate_up + static_cast<size_t>(s) * 2 * kFfn;
    const float* w = weights;
    const int ew_blocks = (s * kFfn + 255) / 256 > 4096 ? 4096 : (s * kFfn + 255) / 256;

    rmsNormRows<<<rowBlocks(s), 256>>>(x, xn, w + kRms1, s, kD, 1e-5f);
    gemm<true>(xn, kD, w + kWq, kD, qkv, kQkvDim, s, kD, kQkvDim, NoEpi{});
    applyRope<<<ew_blocks, 256>>>(qkv, cos, sin, s);
    AttnParams p{};
    p.q_rows = s;
    p.kv_rows = s;
    p.head_dim = kDh;
    p.group = kQHeads / kKvHeads;
    p.causal = 1;
    p.q_stride = p.k_stride = p.v_stride = kQkvDim;
    p.o_stride = kD;
    p.q_head = p.k_head = p.v_head = p.o_head = kDh;
    p.scale = 1.0f / sqrtf(static_cast<float>(kDh));
    attention(qkv, qkv + kD, qkv + kD + kKvDim, attn, p, kQHeads, 1);
    gemm<true>(attn, kD, w + kWo, kD, hidden, kD, s, kD, kD, ResidualEpi{x, kD});
    rmsNormRows<<<rowBlocks(s), 256>>>(hidden, xn, w + kRms2, s, kD, 1e-5f);
    gemm<true>(xn, kD, w + kWgate, kD, gate_up, 2 * kFfn, s, kD, 2 * kFfn, NoEpi{});
    swiglu<<<ew_blocks, 256>>>(gate_up, ffn, s);
    gemm<true>(ffn, kFfn, w + kWdown, kFfn, output, kD, s, kFfn, kD, ResidualEpi{hidden, kD});
    cudaDeviceSynchronize();
    cudaFree(buf);
}

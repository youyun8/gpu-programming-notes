// Softmax Attention Backward (LeetGPU)
// https://leetgpu.com/challenges/softmax-attention-backward
//
// FlashAttention-2 style backward (d <= 128), never materializing M x N:
//   1. rowStats (query-parallel): recompute O_i with an online softmax, store
//      L_i = logsumexp_j(S_ij) and D_i = dO_i . O_i  (= sum_j P_ij dP_ij).
//   2. gradQ (query-parallel):  dQ_i = sum_j dS_ij K_j / sqrt(d)
//   3. gradKV (key-parallel):    dV_j = sum_i P_ij dO_i,  dK_j = sum_i dS_ij Q_i / sqrt(d)
// with P_ij = exp(S_ij - L_i), dS_ij = P_ij (dO_i . V_j - D_i).
// All three kernels share one pattern: a warp owns one row; the block streams
// tiles of 32 "other side" rows through shared memory; lane l evaluates the
// pair with row l of the tile; per-pair weights are broadcast with shuffles
// and each lane accumulates d/32 output columns.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kWarps = 4;
constexpr int kTile = 32;
constexpr int kMaxDim = 128;
constexpr int kPitch = kMaxDim + 1;
constexpr int kCols = kMaxDim / 32;

__device__ __forceinline__ float warpMax(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warpSum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Loads rows [row0, row0 + 32) of up to two N x d matrices into shared tiles (zero padded).
__device__ void loadTiles(const float* a, const float* b, float (*ta)[kPitch], float (*tb)[kPitch], int row0, int rows, int d) {
    for (int i = threadIdx.x; i < kTile * d; i += blockDim.x) {
        const int r = i / d;
        const int c = i % d;
        const bool ok = row0 + r < rows;
        const size_t g = static_cast<size_t>(row0 + r) * d + c;
        ta[r][c] = ok ? a[g] : 0.0f;
        tb[r][c] = ok ? b[g] : 0.0f;
    }
}

__global__ void rowStats(const float* q, const float* k, const float* v, const float* d_o, float* lse, float* delta, int m,
                         int n, int d, float scale) {
    __shared__ float k_t[kTile][kPitch];
    __shared__ float v_t[kTile][kPitch];
    __shared__ float q_s[kWarps][kMaxDim];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int row = blockIdx.x * kWarps + warp;
    const bool active = row < m;
    for (int c = lane; c < d; c += 32) q_s[warp][c] = active ? q[static_cast<size_t>(row) * d + c] * scale : 0.0f;
    float acc[kCols] = {};
    float mx = -FLT_MAX, sum = 0.0f;
    for (int j0 = 0; j0 < n; j0 += kTile) {
        __syncthreads();
        loadTiles(k, v, k_t, v_t, j0, n, d);
        __syncthreads();
        float s = -FLT_MAX;
        if (j0 + lane < n) {
            s = 0.0f;
            for (int c = 0; c < d; ++c) s = fmaf(q_s[warp][c], k_t[lane][c], s);
        }
        const float new_mx = fmaxf(mx, warpMax(s));
        const float corr = expf(mx - new_mx);
        const float p = j0 + lane < n ? expf(s - new_mx) : 0.0f;
        sum = sum * corr + warpSum(p);
        mx = new_mx;
        for (int r = 0; r < kCols; ++r) acc[r] *= corr;
        for (int j = 0; j < min(kTile, n - j0); ++j) {
            const float pj = __shfl_sync(0xffffffffu, p, j);
            for (int r = 0; r < kCols; ++r) {
                const int c = lane + 32 * r;
                if (c < d) acc[r] = fmaf(pj, v_t[j][c], acc[r]);
            }
        }
    }
    float dot = 0.0f;
    if (active) {
        for (int r = 0; r < kCols; ++r) {
            const int c = lane + 32 * r;
            if (c < d) dot += d_o[static_cast<size_t>(row) * d + c] * (acc[r] / sum);
        }
    }
    dot = warpSum(dot);
    if (active && lane == 0) {
        lse[row] = mx + logf(sum);
        delta[row] = dot;
    }
}

__global__ void gradQ(const float* q, const float* k, const float* v, const float* d_o, const float* lse, const float* delta,
                      float* dq, int m, int n, int d, float scale) {
    __shared__ float k_t[kTile][kPitch];
    __shared__ float v_t[kTile][kPitch];
    __shared__ float q_s[kWarps][kMaxDim];
    __shared__ float do_s[kWarps][kMaxDim];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int row = blockIdx.x * kWarps + warp;
    const bool active = row < m;
    for (int c = lane; c < d; c += 32) {
        q_s[warp][c] = active ? q[static_cast<size_t>(row) * d + c] * scale : 0.0f;
        do_s[warp][c] = active ? d_o[static_cast<size_t>(row) * d + c] : 0.0f;
    }
    const float li = active ? lse[row] : 0.0f;
    const float di = active ? delta[row] : 0.0f;
    float acc[kCols] = {};
    for (int j0 = 0; j0 < n; j0 += kTile) {
        __syncthreads();
        loadTiles(k, v, k_t, v_t, j0, n, d);
        __syncthreads();
        float ds = 0.0f;
        if (active && j0 + lane < n) {
            float s = 0.0f, dp = 0.0f;
            for (int c = 0; c < d; ++c) {
                s = fmaf(q_s[warp][c], k_t[lane][c], s);
                dp = fmaf(do_s[warp][c], v_t[lane][c], dp);
            }
            const float p = expf(s - li);
            ds = p * (dp - di) * scale;
        }
        for (int j = 0; j < min(kTile, n - j0); ++j) {
            const float w = __shfl_sync(0xffffffffu, ds, j);
            for (int r = 0; r < kCols; ++r) {
                const int c = lane + 32 * r;
                if (c < d) acc[r] = fmaf(w, k_t[j][c], acc[r]);
            }
        }
    }
    if (active)
        for (int r = 0; r < kCols; ++r) {
            const int c = lane + 32 * r;
            if (c < d) dq[static_cast<size_t>(row) * d + c] = acc[r];
        }
}

__global__ void gradKV(const float* q, const float* k, const float* v, const float* d_o, const float* lse, const float* delta,
                       float* dk, float* dv, int m, int n, int d, float scale) {
    __shared__ float q_t[kTile][kPitch];
    __shared__ float do_t[kTile][kPitch];
    __shared__ float l_t[kTile];
    __shared__ float d_t[kTile];
    __shared__ float k_s[kWarps][kMaxDim];
    __shared__ float v_s[kWarps][kMaxDim];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int key = blockIdx.x * kWarps + warp;
    const bool active = key < n;
    for (int c = lane; c < d; c += 32) {
        k_s[warp][c] = active ? k[static_cast<size_t>(key) * d + c] : 0.0f;
        v_s[warp][c] = active ? v[static_cast<size_t>(key) * d + c] : 0.0f;
    }
    float acc_k[kCols] = {};
    float acc_v[kCols] = {};
    for (int i0 = 0; i0 < m; i0 += kTile) {
        __syncthreads();
        loadTiles(q, d_o, q_t, do_t, i0, m, d);
        for (int i = threadIdx.x; i < kTile; i += blockDim.x) {
            l_t[i] = i0 + i < m ? lse[i0 + i] : 0.0f;
            d_t[i] = i0 + i < m ? delta[i0 + i] : 0.0f;
        }
        __syncthreads();
        float p = 0.0f, ds = 0.0f;
        if (active && i0 + lane < m) {
            float s = 0.0f, dp = 0.0f;
            for (int c = 0; c < d; ++c) {
                s = fmaf(q_t[lane][c], k_s[warp][c], s);
                dp = fmaf(do_t[lane][c], v_s[warp][c], dp);
            }
            p = expf(s * scale - l_t[lane]);
            ds = p * (dp - d_t[lane]) * scale;
        }
        for (int i = 0; i < min(kTile, m - i0); ++i) {
            const float pi = __shfl_sync(0xffffffffu, p, i);
            const float dsi = __shfl_sync(0xffffffffu, ds, i);
            for (int r = 0; r < kCols; ++r) {
                const int c = lane + 32 * r;
                if (c < d) {
                    acc_v[r] = fmaf(pi, do_t[i][c], acc_v[r]);
                    acc_k[r] = fmaf(dsi, q_t[i][c], acc_k[r]);
                }
            }
        }
    }
    if (active)
        for (int r = 0; r < kCols; ++r) {
            const int c = lane + 32 * r;
            if (c < d) {
                dk[static_cast<size_t>(key) * d + c] = acc_k[r];
                dv[static_cast<size_t>(key) * d + c] = acc_v[r];
            }
        }
}

// all pointers are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* dO, float* dQ, float* dK, float* dV,
                      int M, int N, int d) {
    float* stats = nullptr;
    cudaMalloc(&stats, 2 * static_cast<size_t>(M) * sizeof(float));
    float* lse = stats;
    float* delta = stats + M;
    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    rowStats<<<(M + kWarps - 1) / kWarps, kWarps * 32>>>(Q, K, V, dO, lse, delta, M, N, d, scale);
    gradQ<<<(M + kWarps - 1) / kWarps, kWarps * 32>>>(Q, K, V, dO, lse, delta, dQ, M, N, d, scale);
    gradKV<<<(N + kWarps - 1) / kWarps, kWarps * 32>>>(Q, K, V, dO, lse, delta, dK, dV, M, N, d, scale);
    cudaDeviceSynchronize();
    cudaFree(stats);
}

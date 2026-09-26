// Chapter 13: softmax, layer normalization and FlashAttention (FP32, CUDA cores).
//
//   softmaxThreePass    one block per row: max, then sum of exp, then normalize (3 reads of the row)
//   softmaxOnline       one warp per row: max and sum in ONE pass with the online rescaling
//                       (m, z) monoid, then normalize (2 reads)
//   layerNormWelford    one block per row: Welford/Chan mean and variance in one pass
//   flashAttention      O = softmax(Q K^T / sqrt(d)) V without ever storing the N x N scores:
//                       K/V tiles in shared memory, online softmax per query row, optional causal mask
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 13-softmax-attention.cu -o softmax_attention
#include <utility>

#include "check.cuh"

constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kThreads = 256;

__device__ float warpMax(float v) {
    for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(kFullMask, v, m));
    return v;
}
__device__ float warpSum(float v) {
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(kFullMask, v, m);
    return v;
}

// Block-wide all-reduce (every thread gets the result) built on the warp versions.
template <bool kIsMax>
__device__ float blockAllReduce(float v) {
    __shared__ float partial[32];
    __shared__ float result;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, num_warps = blockDim.x / 32;
    v = kIsMax ? warpMax(v) : warpSum(v);
    if (lane == 0) partial[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = lane < num_warps ? partial[lane] : (kIsMax ? -INFINITY : 0.0f);
        t = kIsMax ? warpMax(t) : warpSum(t);
        if (lane == 0) result = t;
    }
    __syncthreads();
    const float r = result;
    __syncthreads();   // partial/result may be reused by the next call
    return r;
}

// ---- Softmax ----------------------------------------------------------------------------------
__global__ void softmaxThreePass(const float* in, float* out, int cols) {
    const float* x = in + static_cast<size_t>(blockIdx.x) * cols;
    float* y = out + static_cast<size_t>(blockIdx.x) * cols;
    float m = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, x[c]);   // pass 1
    m = blockAllReduce<true>(m);
    float z = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) z += __expf(x[c] - m);   // pass 2
    z = blockAllReduce<false>(z);
    const float inv = 1.0f / z;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) y[c] = __expf(x[c] - m) * inv;   // pass 3
}

// The (m, z) monoid of chapter 03: combine two partial (max, sum of exp(x - max)) states.
struct MaxSum {
    float m, z;
};
__device__ __forceinline__ MaxSum combine(MaxSum a, MaxSum b) {
    const float m = fmaxf(a.m, b.m);
    if (m == -INFINITY) return {m, 0.0f};                 // both empty
    return {m, a.z * __expf(a.m - m) + b.z * __expf(b.m - m)};
}

// One warp per row; each lane walks its columns once, updating (m, z) online.
__global__ void softmaxOnline(const float* in, float* out, int rows, int cols) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) return;                              // whole warps exit together
    const float* x = in + static_cast<size_t>(row) * cols;
    float* y = out + static_cast<size_t>(row) * cols;
    MaxSum s{-INFINITY, 0.0f};
    for (int c = lane; c < cols; c += 32) {               // pass 1: max and sum together
        const float v = x[c];
        if (v > s.m) {
            s.z = s.z * __expf(s.m - v) + 1.0f;           // rescale the old sum to the new max
            s.m = v;
        } else {
            s.z += __expf(v - s.m);
        }
    }
    for (int d = 16; d > 0; d >>= 1) {                    // combine the 32 lanes' states (butterfly)
        const MaxSum o{__shfl_xor_sync(kFullMask, s.m, d), __shfl_xor_sync(kFullMask, s.z, d)};
        s = combine(s, o);
    }
    const float inv = 1.0f / s.z;
    for (int c = lane; c < cols; c += 32) y[c] = __expf(x[c] - s.m) * inv;   // pass 2
}

// ---- LayerNorm --------------------------------------------------------------------------------
struct Moments {
    float n, mean, m2;
};
__device__ __forceinline__ Moments combine(Moments a, Moments b) {
    const float n = a.n + b.n;
    if (n == 0.0f) return a;
    const float delta = b.mean - a.mean;
    return {n, a.mean + delta * (b.n / n), a.m2 + b.m2 + delta * delta * (a.n * b.n / n)};
}
__device__ Moments warpAllReduceMoments(Moments v) {
    for (int d = 16; d > 0; d >>= 1) {
        const Moments o{__shfl_xor_sync(kFullMask, v.n, d), __shfl_xor_sync(kFullMask, v.mean, d),
                        __shfl_xor_sync(kFullMask, v.m2, d)};
        v = combine(v, o);
    }
    return v;
}

// y = (x - mean) / sqrt(var + eps) * gamma + beta, one block per row, one pass for the statistics.
__global__ void layerNormWelford(const float* in, const float* gamma, const float* beta, float* out, int cols,
                                 float eps) {
    __shared__ Moments partial[32];
    __shared__ float s_mean, s_rstd;
    const float* x = in + static_cast<size_t>(blockIdx.x) * cols;
    float* y = out + static_cast<size_t>(blockIdx.x) * cols;
    Moments acc{0.0f, 0.0f, 0.0f};
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {   // Welford update, one element at a time
        acc.n += 1.0f;
        const float delta = x[c] - acc.mean;
        acc.mean += delta / acc.n;
        acc.m2 += delta * (x[c] - acc.mean);
    }
    acc = warpAllReduceMoments(acc);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) partial[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        Moments t = lane < blockDim.x / 32 ? partial[lane] : Moments{0.0f, 0.0f, 0.0f};
        t = warpAllReduceMoments(t);
        if (lane == 0) {
            s_mean = t.mean;
            s_rstd = rsqrtf(t.m2 / t.n + eps);
        }
    }
    __syncthreads();
    for (int c = threadIdx.x; c < cols; c += blockDim.x) y[c] = (x[c] - s_mean) * s_rstd * gamma[c] + beta[c];
}

// ---- FlashAttention (forward) -----------------------------------------------------------------
constexpr int kHeadDim = 64;       // d
constexpr int kBlockQ = 16;        // query rows per block (4 warps x 4 rows)
constexpr int kBlockKv = 32;       // keys per K/V tile (one per lane)
constexpr int kAttnThreads = 128;
constexpr int kRowsPerWarp = kBlockQ / (kAttnThreads / 32);
constexpr int kDimsPerLane = kHeadDim / 32;

// Q, K, V, O: N x d, row-major, one head. Each block owns kBlockQ query rows and streams over
// all K/V tiles; the N x N score matrix is never written to memory.
__global__ void __launch_bounds__(kAttnThreads) flashAttention(const float* q, const float* k, const float* v,
                                                               float* o, int n, bool causal) {
    __shared__ float q_s[kBlockQ][kHeadDim];
    __shared__ float k_s[kBlockKv][kHeadDim + 1];   // +1: lane j reads row j: conflict-free
    __shared__ float v_s[kBlockKv][kHeadDim];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int q0 = blockIdx.x * kBlockQ;
    const float scale = rsqrtf(static_cast<float>(kHeadDim));

    for (int i = threadIdx.x; i < kBlockQ * kHeadDim; i += blockDim.x) {
        const int r = i / kHeadDim, c = i % kHeadDim;
        q_s[r][c] = q0 + r < n ? q[static_cast<size_t>(q0 + r) * kHeadDim + c] * scale : 0.0f;
    }
    // Per query row owned by this warp: running max m, running sum l, and the output
    // accumulator, of which each lane holds kDimsPerLane = 2 dimensions.
    float m[kRowsPerWarp], l[kRowsPerWarp], acc[kRowsPerWarp][kDimsPerLane];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
#pragma unroll
        for (int e = 0; e < kDimsPerLane; ++e) acc[r][e] = 0.0f;
    }
    // With a causal mask, key tiles past the block's last query contribute nothing.
    const int kv_end = causal ? min(n, q0 + kBlockQ) : n;
    for (int kv0 = 0; kv0 < kv_end; kv0 += kBlockKv) {
        __syncthreads();                                  // previous tile fully used (and q_s written)
        for (int i = threadIdx.x; i < kBlockKv * kHeadDim; i += blockDim.x) {
            const int r = i / kHeadDim, c = i % kHeadDim;
            const bool in_range = kv0 + r < n;
            k_s[r][c] = in_range ? k[static_cast<size_t>(kv0 + r) * kHeadDim + c] : 0.0f;
            v_s[r][c] = in_range ? v[static_cast<size_t>(kv0 + r) * kHeadDim + c] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            const int qr = warp * kRowsPerWarp + r;       // row inside the block
            const int qi = q0 + qr;                       // global query index
            const int kj = kv0 + lane;                    // this lane's key
            // 1. Score of (qi, kj): a 64-long dot product; q_s is a broadcast, k_s row lane.
            float s = 0.0f;
            for (int c = 0; c < kHeadDim; ++c) s = fmaf(q_s[qr][c], k_s[lane][c], s);
            if (kj >= n || (causal && kj > qi)) s = -INFINITY;
            // 2. Online softmax: new running max, rescale what was accumulated so far.
            const float m_new = fmaxf(m[r], warpMax(s));
            if (m_new == -INFINITY) continue;             // nothing visible yet for this row
            const float p = __expf(s - m_new);            // this lane's unnormalized probability
            const float rescale = __expf(m[r] - m_new);
            l[r] = l[r] * rescale + warpSum(p);
            m[r] = m_new;
            // 3. acc = acc * rescale + P V: lane owns dims (lane + 32 e); p of key j via a shuffle.
#pragma unroll
            for (int e = 0; e < kDimsPerLane; ++e) acc[r][e] *= rescale;
#pragma unroll 8
            for (int j = 0; j < kBlockKv; ++j) {
                const float pj = __shfl_sync(kFullMask, p, j);
#pragma unroll
                for (int e = 0; e < kDimsPerLane; ++e) acc[r][e] = fmaf(pj, v_s[j][lane + 32 * e], acc[r][e]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int qi = q0 + warp * kRowsPerWarp + r;
        if (qi >= n) continue;
#pragma unroll
        for (int e = 0; e < kDimsPerLane; ++e)
            o[static_cast<size_t>(qi) * kHeadDim + lane + 32 * e] = acc[r][e] / l[r];
    }
}

// ---- Checks -----------------------------------------------------------------------------------
std::vector<double> softmaxReference(const std::vector<float>& x, int rows, int cols) {
    std::vector<double> y(x.size());
    for (int r = 0; r < rows; ++r) {
        double m = -1e300, z = 0.0;
        for (int c = 0; c < cols; ++c) m = std::max(m, static_cast<double>(x[r * cols + c]));
        for (int c = 0; c < cols; ++c) z += std::exp(x[r * cols + c] - m);
        for (int c = 0; c < cols; ++c) y[r * cols + c] = std::exp(x[r * cols + c] - m) / z;
    }
    return y;
}

void checkSoftmax(int rows, int cols) {
    const std::vector<float> x = ex::randomVector(static_cast<size_t>(rows) * cols, 1, -8.0f, 8.0f);
    const std::vector<double> ref = softmaxReference(x, rows, cols);
    ex::DeviceArray<float> d_in(x), d_out(x.size());
    char name[64];
    softmaxThreePass<<<rows, kThreads>>>(d_in.ptr, d_out.ptr, cols);
    std::snprintf(name, sizeof(name), "softmaxThreePass %dx%d", rows, cols);
    ex::checkClose(name, d_out.download(), ref, 1e-4, 1e-7);
    softmaxOnline<<<ex::ceilDiv(rows * 32, kThreads), kThreads>>>(d_in.ptr, d_out.ptr, rows, cols);
    std::snprintf(name, sizeof(name), "softmaxOnline %dx%d", rows, cols);
    ex::checkClose(name, d_out.download(), ref, 1e-4, 1e-7);
}

void checkLayerNorm(int rows, int cols) {
    // A large common offset makes the naive E[x^2] - E[x]^2 formula lose most of its digits.
    const std::vector<float> x = ex::randomVector(static_cast<size_t>(rows) * cols, 2, 999.0f, 1001.0f);
    const std::vector<float> gamma = ex::randomVector(cols, 3, 0.5f, 1.5f), beta = ex::randomVector(cols, 4);
    const float eps = 1e-5f;
    std::vector<double> ref(x.size());
    for (int r = 0; r < rows; ++r) {
        double mean = 0.0, var = 0.0;
        for (int c = 0; c < cols; ++c) mean += x[r * cols + c];
        mean /= cols;
        for (int c = 0; c < cols; ++c) var += (x[r * cols + c] - mean) * (x[r * cols + c] - mean);
        var /= cols;
        for (int c = 0; c < cols; ++c)
            ref[r * cols + c] = (x[r * cols + c] - mean) / std::sqrt(var + eps) * gamma[c] + beta[c];
    }
    ex::DeviceArray<float> d_in(x), d_gamma(gamma), d_beta(beta), d_out(x.size());
    layerNormWelford<<<rows, kThreads>>>(d_in.ptr, d_gamma.ptr, d_beta.ptr, d_out.ptr, cols, eps);
    char name[64];
    std::snprintf(name, sizeof(name), "layerNormWelford %dx%d (mean ~1000)", rows, cols);
    ex::checkClose(name, d_out.download(), ref, 2e-3, 2e-3);
}

void checkAttention(int n, bool causal) {
    const size_t count = static_cast<size_t>(n) * kHeadDim;
    const std::vector<float> q = ex::randomVector(count, 5), k = ex::randomVector(count, 6),
                             v = ex::randomVector(count, 7);
    std::vector<double> ref(count, 0.0);
    const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
    for (int i = 0; i < n; ++i) {
        std::vector<double> s(n);
        double m = -1e300, z = 0.0;
        const int visible = causal ? i + 1 : n;
        for (int j = 0; j < visible; ++j) {
            double dot = 0.0;
            for (int c = 0; c < kHeadDim; ++c) dot += static_cast<double>(q[i * kHeadDim + c]) * k[j * kHeadDim + c];
            s[j] = dot * scale;
            m = std::max(m, s[j]);
        }
        for (int j = 0; j < visible; ++j) z += std::exp(s[j] - m);
        for (int j = 0; j < visible; ++j)
            for (int c = 0; c < kHeadDim; ++c) ref[i * kHeadDim + c] += std::exp(s[j] - m) / z * v[j * kHeadDim + c];
    }
    ex::DeviceArray<float> d_q(q), d_k(k), d_v(v), d_o(count);
    flashAttention<<<ex::ceilDiv(n, kBlockQ), kAttnThreads>>>(d_q.ptr, d_k.ptr, d_v.ptr, d_o.ptr, n, causal);
    char name[64];
    std::snprintf(name, sizeof(name), "flashAttention N=%d d=%d%s", n, kHeadDim, causal ? " causal" : "");
    ex::checkClose(name, d_o.download(), ref, 1e-3, 1e-4);
}

void bench() {
    const int rows = 4096, cols = 4096;
    ex::DeviceArray<float> d_in(ex::randomVector(static_cast<size_t>(rows) * cols, 8)), d_out(rows * cols);
    const double bytes = 8.0 * rows * cols;
    ex::reportBandwidth("softmaxThreePass 4096x4096", ex::timeMs([&] {
        softmaxThreePass<<<rows, kThreads>>>(d_in.ptr, d_out.ptr, cols);
    }), bytes);
    ex::reportBandwidth("softmaxOnline 4096x4096", ex::timeMs([&] {
        softmaxOnline<<<rows * 32 / kThreads, kThreads>>>(d_in.ptr, d_out.ptr, rows, cols);
    }), bytes);
    const int n = 4096;
    const size_t count = static_cast<size_t>(n) * kHeadDim;
    ex::DeviceArray<float> d_q(ex::randomVector(count, 9)), d_k(ex::randomVector(count, 10)),
        d_v(ex::randomVector(count, 11)), d_o(count);
    const float ms = ex::timeMs([&] {
        flashAttention<<<n / kBlockQ, kAttnThreads>>>(d_q.ptr, d_k.ptr, d_v.ptr, d_o.ptr, n, false);
    });
    std::printf("%-40s %8.3f ms  %7.2f TFLOP/s\n", "flashAttention N=4096 d=64", ms,
                4.0 * n * n * kHeadDim / (ms * 1e-3) * 1e-12);
}

int main(int argc, char** argv) {
    using Shapes = std::vector<std::pair<int, int>>;
    for (const auto& [r, c] : Shapes{{1, 1}, {3, 31}, {5, 1000}, {2, 4099}}) checkSoftmax(r, c);
    for (const auto& [r, c] : Shapes{{1, 1}, {3, 100}, {2, 3000}}) checkLayerNorm(r, c);
    for (int n : {1, 17, 32, 70}) {
        checkAttention(n, false);
        checkAttention(n, true);
    }
    if (ex::wantBench(argc, argv)) bench();
    return ex::finish("13-softmax-attention");
}

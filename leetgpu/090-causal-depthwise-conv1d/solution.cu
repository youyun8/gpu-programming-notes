// Causal Depthwise Conv1d (LeetGPU)
// https://leetgpu.com/challenges/causal-depthwise-conv1d
//
// out[b, l, d] = bias[d] + sum_k w[d, k] * x[b, l - k, d]   (x = 0 for l - k < 0)
// Channels-last layout: threadIdx.x runs along d, so every tap is a coalesced
// row read; each thread keeps its channel's K <= 8 weights in registers and
// computes several consecutive positions, reusing them.
#include <cuda_runtime.h>

constexpr int kBlockD = 128;
constexpr int kPositionsPerThread = 8;
constexpr int kMaxK = 8;

__global__ void causalDwConv(const float* x, const float* w, const float* bias, float* out, int len, int d, int k) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.z;
    const int l0 = blockIdx.y * kPositionsPerThread;
    if (ch >= d) return;
    float wk[kMaxK];
#pragma unroll
    for (int t = 0; t < kMaxK; ++t) wk[t] = t < k ? w[static_cast<size_t>(ch) * k + t] : 0.0f;
    const float bv = bias[ch];
    const float* xb = x + static_cast<size_t>(b) * len * d;
    float* ob = out + static_cast<size_t>(b) * len * d;
    for (int l = l0; l < min(l0 + kPositionsPerThread, len); ++l) {
        float acc = bv;
        for (int t = k - 1; t >= 0; --t) {  // oldest tap first, like conv1d on the padded input
            const int src = l - t;
            if (src >= 0) acc = fmaf(wk[t], xb[static_cast<size_t>(src) * d + ch], acc);
        }
        ob[static_cast<size_t>(l) * d + ch] = acc;
    }
}

// x, weight, bias, output are device pointers
extern "C" void solve(const float* x, const float* weight, const float* bias, float* output, int B, int L, int D, int K) {
    const dim3 grid((D + kBlockD - 1) / kBlockD, (L + kPositionsPerThread - 1) / kPositionsPerThread, B);
    causalDwConv<<<grid, kBlockD>>>(x, weight, bias, output, L, D, K);
    cudaDeviceSynchronize();
}

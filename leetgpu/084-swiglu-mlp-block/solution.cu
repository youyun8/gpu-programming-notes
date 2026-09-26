// SwiGLU MLP Block (LeetGPU)
// https://leetgpu.com/challenges/swiglu-mlp-block
//
// out = (silu(x Wg) * (x Wu)) Wd.
//   1. dual GEMM: one kernel computes the gate and up tiles together (the x
//      tile is loaded once for both) and applies silu(g) * u in the epilogue,
//      so gate/up are never written to memory; only H (M x d_ffn) is.
//   2. out = H Wd with the same register-blocked SGEMM structure.
// Both kernels: 64 x 64 block tile, 16-wide K slices, 4 x 4 outputs per thread.
#include <cuda_runtime.h>

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreads = 256;

template <bool kDual>
__global__ void __launch_bounds__(kThreads)
gemm(const float* a, const float* b0, const float* b1, float* c, int rows, int inner, int cols) {
    // Register-blocked SGEMM (64 x 64 tile, 4 x 4 outputs per thread). With kDual it computes two
    // products that share A (x W_gate and x W_up) in the same pass.
    __shared__ float a_tile[kTileK][kTileM + 4];
    __shared__ float b0_tile[kTileK][kTileN + 4];
    __shared__ float b1_tile[kDual ? kTileK : 1][kTileN + 4];
    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int row0 = blockIdx.y * kTileM;
    const int col0 = blockIdx.x * kTileN;
    float acc0[4][4] = {};
    float acc1[4][4] = {};
    // Main loop over K in slices of 16: stage A (transposed) and the B panel(s), zero outside.
    for (int k0 = 0; k0 < inner; k0 += kTileK) {
        for (int i = tid; i < kTileM * kTileK; i += kThreads) {
            const int r = i / kTileK;
            const int kk = i % kTileK;
            a_tile[kk][r] = (row0 + r < rows && k0 + kk < inner) ? a[static_cast<size_t>(row0 + r) * inner + k0 + kk] : 0.0f;
        }
        for (int i = tid; i < kTileK * kTileN; i += kThreads) {
            const int kk = i / kTileN;
            const int cc = i % kTileN;
            const bool ok = k0 + kk < inner && col0 + cc < cols;
            const size_t g = static_cast<size_t>(k0 + kk) * cols + col0 + cc;
            b0_tile[kk][cc] = ok ? b0[g] : 0.0f;
            if (kDual) b1_tile[kk][cc] = ok ? b1[g] : 0.0f;
        }
        // Panels complete before anyone reads them.
        __syncthreads();
        // Outer products: each A fragment feeds both accumulators.
#pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            float a_frag[4];
            float b0_frag[4];
            float b1_frag[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a_frag[i] = a_tile[kk][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                b0_frag[j] = b0_tile[kk][tx + 16 * j];
                if (kDual) b1_frag[j] = b1_tile[kk][tx + 16 * j];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc0[i][j] = fmaf(a_frag[i], b0_frag[j], acc0[i][j]);
                    if (kDual) acc1[i][j] = fmaf(a_frag[i], b1_frag[j], acc1[i][j]);
                }
        }
        // Everyone is done with the panels before the next slice overwrites them.
        __syncthreads();
    }
    // Epilogue: fused SwiGLU silu(gate) * up for the first GEMM, plain store for the second.
    for (int i = 0; i < 4; ++i) {
        const int r = row0 + ty + 16 * i;
        if (r >= rows) continue;
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + tx + 16 * j;
            if (col >= cols) continue;
            float v = acc0[i][j];
            if (kDual) v = v / (1.0f + expf(-v)) * acc1[i][j];  // silu(gate) * up
            c[static_cast<size_t>(r) * cols + col] = v;
        }
    }
}

// x, W_gate, W_up, W_down, output are device pointers
extern "C" void solve(const float* x, const float* W_gate, const float* W_up, const float* W_down, float* output, int M,
                      int d_model, int d_ffn) {
    // hidden = silu(x W_gate) * (x W_up) in one kernel, then output = hidden W_down.
    float* hidden = nullptr;
    cudaMalloc(&hidden, static_cast<size_t>(M) * d_ffn * sizeof(float));
    gemm<true><<<dim3((d_ffn + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM), kThreads>>>(x, W_gate, W_up, hidden, M, d_model, d_ffn);
    gemm<false><<<dim3((d_model + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM), kThreads>>>(hidden, W_down, nullptr, output, M, d_ffn, d_model);
    cudaDeviceSynchronize();
    cudaFree(hidden);
}

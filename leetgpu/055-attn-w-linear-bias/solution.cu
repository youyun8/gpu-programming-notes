// Attention with Linear Biases (ALiBi) (LeetGPU)
// https://leetgpu.com/challenges/attention-with-linear-biases
//
// out = softmax(Q K^T / sqrt(d) + alpha * (i - j)) V, M, N <= 2048, d <= 1024.
// d up to 1024 is too large to keep K/V tiles resident per query warp, so this
// uses three well-understood passes over an M x N score buffer (<= 16 MB):
//   1. S = Q K^T * scale + alpha (i - j)   (64 x 64 register-blocked "NT" SGEMM
//                                             with the bias fused in the epilogue)
//   2. row softmax of S in place            (one warp per row)
//   3. out = S V                            (64 x 64 register-blocked "NN" SGEMM)
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreads = 256;

// C[i][j] = sum_k A[i][k] * B'[k][j], where B' = B^T (kTransB) or B.
// A: rows x inner (row-major). B: cols x inner if kTransB, else inner x cols.
template <bool kTransB, bool kAlibi>
__global__ void __launch_bounds__(kThreads)
sgemm(const float* a, const float* b, float* c, int rows, int inner, int cols, float scale, float alpha) {
    __shared__ float a_tile[kTileK][kTileM + 4];
    __shared__ float b_tile[kTileK][kTileN + 4];
    // 16 x 16 threads, each owning a 4 x 4 patch of C (rows ty + 16i, columns tx + 16j).
    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int row0 = blockIdx.y * kTileM;
    const int col0 = blockIdx.x * kTileN;
    float acc[4][4] = {};
    // Main loop over K in slices of 16: stage A (transposed) and B (or B^T), zero outside.
    for (int k0 = 0; k0 < inner; k0 += kTileK) {
        for (int i = tid; i < kTileM * kTileK; i += kThreads) {
            const int r = i / kTileK;
            const int kk = i % kTileK;
            a_tile[kk][r] = (row0 + r < rows && k0 + kk < inner) ? a[static_cast<size_t>(row0 + r) * inner + k0 + kk] : 0.0f;
        }
        if (kTransB) {
            for (int i = tid; i < kTileK * kTileN; i += kThreads) {
                const int cc = i / kTileK;  // k fastest: coalesced reads of B rows
                const int kk = i % kTileK;
                b_tile[kk][cc] = (col0 + cc < cols && k0 + kk < inner) ? b[static_cast<size_t>(col0 + cc) * inner + k0 + kk] : 0.0f;
            }
        } else {
            for (int i = tid; i < kTileK * kTileN; i += kThreads) {
                const int kk = i / kTileN;
                const int cc = i % kTileN;
                b_tile[kk][cc] = (k0 + kk < inner && col0 + cc < cols) ? b[static_cast<size_t>(k0 + kk) * cols + col0 + cc] : 0.0f;
            }
        }
        // Panels complete before anyone reads them.
        __syncthreads();
        // Outer products: 4 + 4 shared loads feed 16 FMAs.
#pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            float a_frag[4];
            float b_frag[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a_frag[i] = a_tile[kk][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b_frag[j] = b_tile[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
        }
        // Everyone is done with the panels before the next slice overwrites them.
        __syncthreads();
    }
    // Epilogue: for the score GEMM, scale and add the ALiBi bias alpha * (i - j).
    for (int i = 0; i < 4; ++i) {
        const int r = row0 + ty + 16 * i;
        if (r >= rows) continue;
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + tx + 16 * j;
            if (col >= cols) continue;
            float v = acc[i][j];
            if (kAlibi) v = v * scale + alpha * static_cast<float>(r - col);
            c[static_cast<size_t>(r) * cols + col] = v;
        }
    }
}

__global__ void rowSoftmax(float* s, int m, int n) {
    // In-place row softmax, one warp per row: max, then exp and sum, then normalize.
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= m) return;
    float* p = s + static_cast<size_t>(row) * n;
    float mx = -FLT_MAX;
    for (int j = lane; j < n; j += 32) mx = fmaxf(mx, p[j]);
    for (int offset = 16; offset > 0; offset >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, offset));
    float sum = 0.0f;
    for (int j = lane; j < n; j += 32) {
        const float e = expf(p[j] - mx);
        p[j] = e;
        sum += e;
    }
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    const float inv = 1.0f / sum;
    for (int j = lane; j < n; j += 32) p[j] *= inv;
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d, float alpha) {
    // Unfused: scores = Q K^T / sqrt(d) + bias (M x N buffer), row softmax, then output = P V.
    float* scores = nullptr;
    cudaMalloc(&scores, static_cast<size_t>(M) * N * sizeof(float));
    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    sgemm<true, true><<<dim3((N + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM), kThreads>>>(Q, K, scores, M, d, N, scale, alpha);
    rowSoftmax<<<(M + 7) / 8, 256>>>(scores, M, N);
    sgemm<false, false><<<dim3((d + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM), kThreads>>>(scores, V, output, M, N, d, 1.0f, 0.0f);
    cudaDeviceSynchronize();
    cudaFree(scores);
}

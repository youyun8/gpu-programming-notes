// Lower Triangular Matrix Multiplication (Tensara)
// https://tensara.org/problems/lower-trig-matmul
//
// C = lower(A) * lower(B) for N x N matrices. The product of two lower-triangular
// matrices is lower-triangular, and C[i][j] only needs k between min(i, j) and
// max(i, j). The register-blocked SGEMM is restricted accordingly:
//   - output tiles entirely outside the triangle are written as zeros without
//     touching A or B;
//   - the K loop of the other tiles only visits [col0, row0 + kTileM), roughly a third of
//     the dense FLOPs overall;
//   - loads mask the opposite triangle (the reference applies tril/triu too).
#include <cuda_runtime.h>

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreads = 256;

__global__ void __launch_bounds__(kThreads) triMatmul(const float* a, const float* b, float* c, int n) {
    __shared__ float a_tile[kTileK][kTileM + 4];
    __shared__ float b_tile[kTileK][kTileN + 4];
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int row0 = blockIdx.y * kTileM, col0 = blockIdx.x * kTileN;
    float acc[4][4] = {};
    if (!(col0 > row0 + kTileM - 1)) {
        const int k_begin = (col0) / kTileK * kTileK;
        const int k_end = min(n, row0 + kTileM);
        for (int k0 = k_begin; k0 < k_end; k0 += kTileK) {
            for (int i = tid; i < kTileM * kTileK; i += kThreads) {
                const int r = i / kTileK, kk = i % kTileK;
                const int gr = row0 + r, kk_g = k0 + kk;
                a_tile[kk][r] = (gr < n && kk_g < n && kk_g <= gr) ? a[static_cast<size_t>(gr) * n + kk_g] : 0.0f;
            }
            for (int i = tid; i < kTileK * kTileN; i += kThreads) {
                const int kk = i / kTileN, cc = i % kTileN;
                const int kk_g = k0 + kk, gc = col0 + cc;
                b_tile[kk][cc] = (kk_g < n && gc < n && gc <= kk_g) ? b[static_cast<size_t>(kk_g) * n + gc] : 0.0f;
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
    }
    for (int i = 0; i < 4; ++i) {
        const int r = row0 + ty + 16 * i;
        if (r >= n) continue;
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + tx + 16 * j;
            if (col < n) c[static_cast<size_t>(r) * n + col] = (col <= r) ? acc[i][j] : 0.0f;
        }
    }
}

// input_a, input_b, output_c are device pointers
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n) {
    const int ni = static_cast<int>(n);
    const dim3 grid((ni + kTileN - 1) / kTileN, (ni + kTileM - 1) / kTileM);
    triMatmul<<<grid, kThreads>>>(input_a, input_b, output_c, ni);
}

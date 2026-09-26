// LoRA Linear (LeetGPU)
// https://leetgpu.com/challenges/lora-linear
//
// out = x W^T + s (x A^T) B^T. Two kernels:
//   1. hidden = s * x A^T               (batch x rank, tiny)
//   2. out = [x | hidden] [W | B]^T     one GEMM over a concatenated K
//      dimension (d_in + rank): the base and the low-rank path accumulate
//      into the same registers, so the output is written exactly once.
// Both are "NT" GEMMs (the right operand is stored row-major as N x K), using
// a 64 x 64 register-blocked tile.
#include <cuda_runtime.h>

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreads = 256;

// c = scale * ([a0 | a1] [b0 | b1]^T); the second segment is optional (k1 = 0).
__global__ void __launch_bounds__(kThreads)
gemmNtConcat(const float* a0, const float* b0, int k0_len, const float* a1, const float* b1, int k1_len, float* c,
             int rows, int cols, float scale) {
    __shared__ float a_tile[kTileK][kTileM + 4];
    __shared__ float b_tile[kTileK][kTileN + 4];
    // 16 x 16 threads, each owning a 4 x 4 patch of C (rows ty + 16i, columns tx + 16j).
    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int row0 = blockIdx.y * kTileM;
    const int col0 = blockIdx.x * kTileN;
    float acc[4][4] = {};
    // Accumulate both segments of the concatenated reduction into the same registers.
    for (int seg = 0; seg < 2; ++seg) {
        const float* a = seg == 0 ? a0 : a1;
        const float* b = seg == 0 ? b0 : b1;
        const int inner = seg == 0 ? k0_len : k1_len;
        for (int k0 = 0; k0 < inner; k0 += kTileK) {
            // Stage A (transposed) and B, which is stored (cols x inner) and transposed while loading.
            for (int i = tid; i < kTileM * kTileK; i += kThreads) {
                const int r = i / kTileK;
                const int kk = i % kTileK;
                a_tile[kk][r] = (row0 + r < rows && k0 + kk < inner) ? a[static_cast<size_t>(row0 + r) * inner + k0 + kk] : 0.0f;
            }
            for (int i = tid; i < kTileK * kTileN; i += kThreads) {
                const int cc = i / kTileK;
                const int kk = i % kTileK;
                b_tile[kk][cc] = (col0 + cc < cols && k0 + kk < inner) ? b[static_cast<size_t>(col0 + cc) * inner + k0 + kk] : 0.0f;
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
    }
    // Scale and store in bounds.
    for (int i = 0; i < 4; ++i) {
        const int r = row0 + ty + 16 * i;
        if (r >= rows) continue;
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + tx + 16 * j;
            if (col < cols) c[static_cast<size_t>(r) * cols + col] = scale * acc[i][j];
        }
    }
}

// x, W, A, B, output are device pointers
extern "C" void solve(const float* x, const float* W, const float* A, const float* B, float* output, int batch, int d_in,
                      int d_out, int rank, float lora_scale) {
    // hidden = scale * x A^T (batch x rank), then output = [x | hidden] [W | B]^T:
    // the base projection and the LoRA update in one GEMM, no extra pass over the output.
    float* hidden = nullptr;
    cudaMalloc(&hidden, static_cast<size_t>(batch) * rank * sizeof(float));
    gemmNtConcat<<<dim3((rank + kTileN - 1) / kTileN, (batch + kTileM - 1) / kTileM), kThreads>>>(
        x, A, d_in, nullptr, nullptr, 0, hidden, batch, rank, lora_scale);
    gemmNtConcat<<<dim3((d_out + kTileN - 1) / kTileN, (batch + kTileM - 1) / kTileM), kThreads>>>(
        x, W, d_in, hidden, B, rank, output, batch, d_out, 1.0f);
    cudaDeviceSynchronize();
    cudaFree(hidden);
}

// ViT Patch Embedding (LeetGPU)
// https://leetgpu.com/challenges/vision-transformer-patch-embedding
//
// tokens = patches (B*N x CPP) * W^T (CPP x D) + bias; row 0 of every image is
// cls + pos[0], row n+1 is token n + pos[n+1].
// The patch matrix is never materialized: the GEMM's A-tile loader gathers
// patch pixels straight from the NCHW image (implicit im2col, since stride ==
// kernel == P). W is D x CPP row-major, i.e. an "NT" GEMM. Bias and the
// positional embedding are fused into the epilogue; a tiny kernel writes the
// CLS rows.
#include <cuda_runtime.h>

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreads = 256;

struct PatchGeom {
    int c, h, w, p, gw, n, cpp, d;
};

__device__ __forceinline__ float patchPixel(const float* img, const PatchGeom& g, int row, int k) {
    const int b = row / g.n;
    const int n = row % g.n;
    const int py = n / g.gw, px = n % g.gw;
    const int ch = k / (g.p * g.p);
    const int i = (k / g.p) % g.p;
    const int j = k % g.p;
    return img[((static_cast<size_t>(b) * g.c + ch) * g.h + py * g.p + i) * g.w + px * g.p + j];
}

__global__ void __launch_bounds__(kThreads)
patchGemm(const float* img, const float* weight, const float* bias, const float* pos, float* out, PatchGeom g, int rows) {
    __shared__ float a_tile[kTileK][kTileM + 4];
    __shared__ float b_tile[kTileK][kTileN + 4];
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int row0 = blockIdx.y * kTileM, col0 = blockIdx.x * kTileN;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < g.cpp; k0 += kTileK) {
        for (int i = tid; i < kTileM * kTileK; i += kThreads) {
            const int r = i / kTileK, kk = i % kTileK;
            a_tile[kk][r] = (row0 + r < rows && k0 + kk < g.cpp) ? patchPixel(img, g, row0 + r, k0 + kk) : 0.0f;
        }
        for (int i = tid; i < kTileK * kTileN; i += kThreads) {
            const int cc = i / kTileK, kk = i % kTileK;
            b_tile[kk][cc] = (col0 + cc < g.d && k0 + kk < g.cpp) ? weight[static_cast<size_t>(col0 + cc) * g.cpp + k0 + kk] : 0.0f;
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
        const int b = r / g.n, n = r % g.n;
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + tx + 16 * j;
            if (col >= g.d) continue;
            out[(static_cast<size_t>(b) * (g.n + 1) + n + 1) * g.d + col] =
                acc[i][j] + bias[col] + pos[static_cast<size_t>(n + 1) * g.d + col];
        }
    }
}

__global__ void clsRows(const float* cls, const float* pos, float* out, int batch, int n, int d) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= d) return;
    const float v = cls[col] + pos[col];
    for (int b = 0; b < batch; ++b) out[static_cast<size_t>(b) * (n + 1) * d + col] = v;
}

// all pointers are device pointers
extern "C" void solve(const float* images, const float* patch_weight, const float* patch_bias, const float* cls_token,
                      const float* pos_embed, float* output, int B, int C, int H, int W, int P, int D) {
    PatchGeom g{C, H, W, P, W / P, (H / P) * (W / P), C * P * P, D};
    const int rows = B * g.n;
    patchGemm<<<dim3((D + kTileN - 1) / kTileN, (rows + kTileM - 1) / kTileM), kThreads>>>(images, patch_weight, patch_bias,
                                                                                            pos_embed, output, g, rows);
    clsRows<<<(D + 255) / 256, 256>>>(cls_token, pos_embed, output, B, g.n, D);
    cudaDeviceSynchronize();
}

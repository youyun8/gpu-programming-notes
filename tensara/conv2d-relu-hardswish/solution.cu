// Conv2D + ReLU + HardSwish (Tensara)
// https://tensara.org/problems/conv2d-relu-hardswish
//
// "Same" 2D convolution followed by ReLU and HardSwish, fused: the activation
// is applied to the accumulator before the single store.
#include <cuda_runtime.h>

constexpr int kTileX = 32;
constexpr int kTileY = 32;
constexpr int kBlockY = 8;
constexpr int kRowsPerThread = kTileY / kBlockY;
constexpr int kBand = 8;            // kernel rows staged per pass
constexpr int kMaxKw = 127;
constexpr int kWinCols = kTileX + kMaxKw - 1;

// Output tile 32 x 32 (4 rows per thread). The kernel is walked in bands of 8
// rows: per band the block stages those kernel rows and the input window they
// touch ((32 + 8 - 1) x (32 + Kw - 1), zero padded) in shared memory. This
// bounds shared memory for kernels up to 127 x 127 while still reading each
// input element from DRAM about once per band.
template <class Epi>
__global__ void conv2dSame(const float* __restrict__ in, const float* __restrict__ w, float* __restrict__ out, int h, int wd,
                           int kh, int kw, Epi epi) {
    __shared__ float s_w[kBand][kMaxKw];
    __shared__ float s_in[kTileY + kBand - 1][kWinCols];
    const int row0 = blockIdx.y * kTileY, col0 = blockIdx.x * kTileX;
    const int ph = kh / 2, pw = kw / 2;
    const int win_cols = kTileX + kw - 1;
    const int tid = threadIdx.y * kTileX + threadIdx.x;
    float acc[kRowsPerThread] = {};
    for (int b0 = 0; b0 < kh; b0 += kBand) {
        const int band = min(kBand, kh - b0);
        const int win_rows = kTileY + band - 1;
        __syncthreads();
        for (int i = tid; i < band * kw; i += kTileX * kBlockY) s_w[i / kw][i % kw] = w[(b0 + i / kw) * kw + i % kw];
        for (int i = tid; i < win_rows * win_cols; i += kTileX * kBlockY) {
            const int r = i / win_cols, c = i % win_cols;
            const int gr = row0 + r + b0 - ph, gc = col0 + c - pw;
            s_in[r][c] = (gr >= 0 && gr < h && gc >= 0 && gc < wd) ? in[static_cast<size_t>(gr) * wd + gc] : 0.0f;
        }
        __syncthreads();
        for (int kr = 0; kr < band; ++kr)
            for (int kc = 0; kc < kw; ++kc) {
                const float wv = s_w[kr][kc];
#pragma unroll
                for (int r = 0; r < kRowsPerThread; ++r) acc[r] = fmaf(s_in[threadIdx.y + r * kBlockY + kr][threadIdx.x + kc], wv, acc[r]);
            }
    }
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        const int orow = row0 + threadIdx.y + r * kBlockY, ocol = col0 + threadIdx.x;
        if (orow < h && ocol < wd) out[static_cast<size_t>(orow) * wd + ocol] = epi(acc[r]);
    }
}

struct ReluHardSwish {
    __device__ float operator()(float v) const {
        const float r = fmaxf(v, 0.0f);
        return r * fminf(fmaxf(r + 3.0f, 0.0f), 6.0f) / 6.0f;
    }
};

// image, kernel, output are device pointers
extern "C" void solution(const float* image, const float* kernel, float* output, size_t H, size_t W, size_t Kh, size_t Kw) {
    const dim3 grid(static_cast<unsigned>((W + kTileX - 1) / kTileX), static_cast<unsigned>((H + kTileY - 1) / kTileY));
    conv2dSame<ReluHardSwish><<<grid, dim3(kTileX, kBlockY)>>>(image, kernel, output, static_cast<int>(H), static_cast<int>(W), static_cast<int>(Kh), static_cast<int>(Kw), ReluHardSwish{});
}

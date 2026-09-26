// 2D Convolution (Tensara)
// https://tensara.org/problems/conv-2d
//
// "Same" 2D convolution (cross-correlation) with zero padding, odd kernels of
// up to 127 x 127.
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
    // Shared staging for one band of kernel rows and the input window it touches.
    __shared__ float s_w[kBand][kMaxKw];
    __shared__ float s_in[kTileY + kBand - 1][kWinCols];
    // This block's 32 x 32 output tile; each thread computes 4 rows (threadIdx.y + 8r).
    const int row0 = blockIdx.y * kTileY, col0 = blockIdx.x * kTileX;
    const int ph = kh / 2, pw = kw / 2;
    const int win_cols = kTileX + kw - 1;
    const int tid = threadIdx.y * kTileX + threadIdx.x;
    float acc[kRowsPerThread] = {};
    // Walk the kernel in bands of 8 rows so shared memory is bounded for kernels up to 127 x 127.
    for (int b0 = 0; b0 < kh; b0 += kBand) {
        const int band = min(kBand, kh - b0);
        const int win_rows = kTileY + band - 1;
        // Do not overwrite the previous band while other threads may still read it.
        __syncthreads();
        // Stage the band's kernel rows...
        for (int i = tid; i < band * kw; i += kTileX * kBlockY) s_w[i / kw][i % kw] = w[(b0 + i / kw) * kw + i % kw];
        // ...and the (32 + band - 1) x (32 + kw - 1) input window, zero-filled outside the image.
        for (int i = tid; i < win_rows * win_cols; i += kTileX * kBlockY) {
            const int r = i / win_cols, c = i % win_cols;
            const int gr = row0 + r + b0 - ph, gc = col0 + c - pw;
            s_in[r][c] = (gr >= 0 && gr < h && gc >= 0 && gc < wd) ? in[static_cast<size_t>(gr) * wd + gc] : 0.0f;
        }
        // Staged data visible to all threads.
        __syncthreads();
        // Accumulate: each weight is a broadcast; lanes read consecutive window columns (conflict-free).
        for (int kr = 0; kr < band; ++kr)
            for (int kc = 0; kc < kw; ++kc) {
                const float wv = s_w[kr][kc];
#pragma unroll
                for (int r = 0; r < kRowsPerThread; ++r) acc[r] = fmaf(s_in[threadIdx.y + r * kBlockY + kr][threadIdx.x + kc], wv, acc[r]);
            }
    }
    // Apply the epilogue (identity here) and store the 4 outputs in bounds.
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        const int orow = row0 + threadIdx.y + r * kBlockY, ocol = col0 + threadIdx.x;
        if (orow < h && ocol < wd) out[static_cast<size_t>(orow) * wd + ocol] = epi(acc[r]);
    }
}

// Plain convolution: no activation.
struct Identity {
    __device__ float operator()(float v) const { return v; }
};

// A, B, C are device pointers
extern "C" void solution(const float* A, const float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw) {
    // One 32 x 8 thread block per 32 x 32 output tile.
    const dim3 grid(static_cast<unsigned>((W + kTileX - 1) / kTileX), static_cast<unsigned>((H + kTileY - 1) / kTileY));
    conv2dSame<Identity><<<grid, dim3(kTileX, kBlockY)>>>(A, B, C, static_cast<int>(H), static_cast<int>(W), static_cast<int>(Kh), static_cast<int>(Kw), Identity{});
}

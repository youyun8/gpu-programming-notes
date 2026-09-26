// RoPE Embedding (LeetGPU)
// https://leetgpu.com/challenges/rotary-positional-embedding
//
// out = q * cos + rotate_half(q) * sin, rotate_half([x1, x2]) = [-x2, x1].
// One thread per element pair (j, j + D/2) of a row: both outputs need the
// same two inputs, so each input is read exactly once.
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;

__global__ void ropeKernel(const float* q, const float* cos_t, const float* sin_t, float* out, int m, int d) {
    // One thread per (row, j) pair with j < d/2 (grid-stride): element j is rotated
    // together with element j + d/2 ("rotate half" convention).
    const int half_d = d / 2;
    const long long total = static_cast<long long>(m) * half_d;
    for (long long t = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; t < total;
         t += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long row = t / half_d;
        const int j = static_cast<int>(t % half_d);
        const size_t lo = row * d + j;
        const size_t hi = lo + half_d;
        // 2-D rotation by the per-position angle: (x1, x2) -> (x1 cos - x2 sin, x2 cos + x1 sin).
        const float x1 = q[lo];
        const float x2 = q[hi];
        out[lo] = x1 * cos_t[lo] - x2 * sin_t[lo];
        out[hi] = x2 * cos_t[hi] + x1 * sin_t[hi];
    }
}

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
    // One thread per pair, capped grid.
    const long long total = static_cast<long long>(M) * (D / 2);
    long long blocks = (total + kBlockSize - 1) / kBlockSize;
    blocks = blocks > 65535 ? 65535 : (blocks < 1 ? 1 : blocks);
    ropeKernel<<<static_cast<int>(blocks), kBlockSize>>>(Q, cos, sin, output, M, D);
    cudaDeviceSynchronize();
}

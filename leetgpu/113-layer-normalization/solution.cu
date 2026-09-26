// Layer Normalization (LeetGPU)
// https://leetgpu.com/challenges/layer-normalization
//
// Per-row mean / variance over C features, then weight * xhat + bias.
// One warp per row (rows are independent and C <= 4096): pass 1 sums, pass 2
// accumulates centered squares (two-pass variance, no cancellation), pass 3
// writes. The row stays in L1 between passes, so DRAM traffic is ~1 read + 1 write.
#include <cuda_runtime.h>

constexpr int kWarpsPerBlock = 8;

__global__ void layerNorm(const float* x, const float* w, const float* b, float* y, int n, int c, float eps) {
    // One warp per row.
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * kWarpsPerBlock + threadIdx.x / 32;
    if (row >= n) return;
    const float* xr = x + static_cast<size_t>(row) * c;
    // Mean: lanes stride the row (coalesced), then a butterfly sum.
    float sum = 0.0f;
    for (int j = lane; j < c; j += 32) sum += xr[j];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float mean = sum / c;
    // Centered variance (second read of the row, from cache; no E[x^2] - E[x]^2 cancellation).
    float sq = 0.0f;
    for (int j = lane; j < c; j += 32) {
        const float diff = xr[j] - mean;
        sq += diff * diff;
    }
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    const float rstd = rsqrtf(sq / c + eps);
    // Normalize and apply the affine parameters.
    float* yr = y + static_cast<size_t>(row) * c;
    for (int j = lane; j < c; j += 32) yr[j] = w[j] * ((xr[j] - mean) * rstd) + b[j];
}

// input, weight, bias, output are device pointers
extern "C" void solve(const float* input, const float* weight, const float* bias, float* output, int N, int C, float eps) {
    layerNorm<<<(N + kWarpsPerBlock - 1) / kWarpsPerBlock, kWarpsPerBlock * 32>>>(input, weight, bias, output, N, C, eps);
    cudaDeviceSynchronize();
}

// Batch Normalization (LeetGPU)
// https://leetgpu.com/challenges/batch-normalization
//
// Per-channel statistics over the batch (column reduction of an N x C
// row-major matrix), then an elementwise normalize.
//   channelStats: a 32 x 8 block owns 32 channels; threadIdx.x indexes the
//     channel (coalesced row reads), threadIdx.y strides the rows. Mean and
//     M2 are accumulated with Welford's update in fp64 and merged across the
//     8 row-groups (Chan's parallel formula).
//   normalize: y = gamma * (x - mean) * rstd + beta.
#include <cuda_runtime.h>

constexpr int kChannels = 32;
constexpr int kRowGroups = 8;

__global__ void channelStats(const float* x, float* mean_out, float* rstd_out, int n, int c, float eps) {
    // Per channel: 8 row groups each keep a Welford (count, mean, M2) in double.
    __shared__ double s_mean[kRowGroups][kChannels];
    __shared__ double s_m2[kRowGroups][kChannels];
    __shared__ int s_count[kRowGroups];
    // threadIdx.x = channel (32 consecutive channels: coalesced), threadIdx.y = row group.
    const int ch = blockIdx.x * kChannels + threadIdx.x;
    double mean = 0.0, m2 = 0.0;
    int count = 0;
    // Welford update over this group's rows (numerically stable single pass).
    if (ch < c) {
        for (int r = threadIdx.y; r < n; r += kRowGroups) {
            const double v = x[static_cast<size_t>(r) * c + ch];
            ++count;
            const double delta = v - mean;
            mean += delta / count;
            m2 += delta * (v - mean);
        }
    }
    // Publish the partial statistics.
    s_mean[threadIdx.y][threadIdx.x] = mean;
    s_m2[threadIdx.y][threadIdx.x] = m2;
    if (threadIdx.x == 0) s_count[threadIdx.y] = count;
    __syncthreads();
    // Merge the 8 partials with Chan's formula; store the mean and 1 / sqrt(var + eps) (biased variance).
    if (threadIdx.y == 0 && ch < c) {
        double m = 0.0, s = 0.0;
        int total = 0;
        for (int g = 0; g < kRowGroups; ++g) {
            const int cnt = s_count[g];
            if (cnt == 0) continue;
            const double delta = s_mean[g][threadIdx.x] - m;
            const int merged = total + cnt;
            m += delta * cnt / merged;
            s += s_m2[g][threadIdx.x] + delta * delta * static_cast<double>(total) * cnt / merged;
            total = merged;
        }
        mean_out[ch] = static_cast<float>(m);
        rstd_out[ch] = static_cast<float>(1.0 / sqrt(s / n + eps));
    }
}

__global__ void normalize(const float* x, const float* gamma, const float* beta, const float* mean, const float* rstd,
                          float* y, int n, int c) {
    // Elementwise normalization and affine transform (grid-stride).
    const size_t total = static_cast<size_t>(n) * c;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int ch = static_cast<int>(i % c);
        y[i] = gamma[ch] * ((x[i] - mean[ch]) * rstd[ch]) + beta[ch];
    }
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int N, int C, float eps) {
    // Per-channel mean and rstd, then the normalization pass.
    float* stats = nullptr;
    cudaMalloc(&stats, 2 * static_cast<size_t>(C) * sizeof(float));
    channelStats<<<(C + kChannels - 1) / kChannels, dim3(kChannels, kRowGroups)>>>(input, stats, stats + C, N, C, eps);
    const size_t total = static_cast<size_t>(N) * C;
    int blocks = static_cast<int>((total + 255) / 256);
    blocks = blocks > 4096 ? 4096 : blocks;
    normalize<<<blocks, 256>>>(input, gamma, beta, stats, stats + C, output, N, C);
    cudaDeviceSynchronize();
    cudaFree(stats);
}

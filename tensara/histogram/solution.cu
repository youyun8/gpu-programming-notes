// Histogram (Tensara)
// https://tensara.org/problems/histogram
//
// bin = (int)clamp(pixel, 0, num_bins - 1); counts are returned as floats.
// Privatized histogram: each block counts into shared memory (fast shared
// atomics, contention stays inside the block), then adds its non-zero bins to
// the global histogram, which is zeroed first.
#include <cuda_runtime.h>

constexpr int kThreads = 256;
constexpr int kMaxSharedBins = 8192;

__global__ void zeroBins(float* hist, int bins) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < bins; i += gridDim.x * blockDim.x) hist[i] = 0.0f;
}

__global__ void histogramKernel(const float* __restrict__ img, float* hist, size_t n, int bins) {
    __shared__ unsigned int s_hist[kMaxSharedBins];
    const bool shared = bins <= kMaxSharedBins;
    if (shared)
        for (int b = threadIdx.x; b < bins; b += kThreads) s_hist[b] = 0;
    __syncthreads();
    const float top = static_cast<float>(bins - 1);
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int b = static_cast<int>(fminf(fmaxf(img[i], 0.0f), top));
        if (shared) atomicAdd(&s_hist[b], 1u);
        else atomicAdd(&hist[b], 1.0f);
    }
    __syncthreads();
    if (shared)
        for (int b = threadIdx.x; b < bins; b += kThreads)
            if (s_hist[b]) atomicAdd(&hist[b], static_cast<float>(s_hist[b]));
}

// image, histogram are device pointers
extern "C" void solution(const float* image, int num_bins, float* histogram, size_t height, size_t width) {
    const size_t n = height * width;
    zeroBins<<<(num_bins + kThreads - 1) / kThreads, kThreads>>>(histogram, num_bins);
    size_t blocks = (n + kThreads - 1) / kThreads;
    blocks = blocks > 1024 ? 1024 : (blocks < 1 ? 1 : blocks);
    histogramKernel<<<static_cast<unsigned>(blocks), kThreads>>>(image, histogram, n, num_bins);
}

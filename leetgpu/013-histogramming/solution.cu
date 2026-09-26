// Histogramming (LeetGPU)
// https://leetgpu.com/challenges/histogramming
//
// Privatized histogram: each block counts into a shared-memory copy with fast
// shared atomics, then merges its non-zero bins into the global histogram.
// This turns N global atomics (heavily contended on few bins) into
// num_blocks * num_bins.
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kMaxBins = 1024;
constexpr int kMaxBlocks = 1024;

__global__ void histogramKernel(const int* input, int* histogram, int n, int num_bins) {
    // Per-block private histogram in shared memory.
    __shared__ int s_hist[kMaxBins];
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x) s_hist[b] = 0;
    __syncthreads();

    // Grid-stride count with shared-memory atomics (contention stays inside the block).
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int value = input[i];
        if (value >= 0 && value < num_bins) atomicAdd(&s_hist[value], 1);
    }
    __syncthreads();

    // Merge the non-zero bins into the global histogram.
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
        const int count = s_hist[b];
        if (count > 0) atomicAdd(&histogram[b], count);
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    // The output is accumulated with atomics, so zero it first; at most 1024 blocks.
    cudaMemset(histogram, 0, num_bins * sizeof(int));
    int num_blocks = (N + kBlockSize - 1) / kBlockSize;
    num_blocks = num_blocks > kMaxBlocks ? kMaxBlocks : num_blocks;
    histogramKernel<<<num_blocks, kBlockSize>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}

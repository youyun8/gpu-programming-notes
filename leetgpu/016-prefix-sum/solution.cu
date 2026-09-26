// Prefix Sum (LeetGPU)
// https://leetgpu.com/challenges/prefix-sum
//
// Reduce-then-scan, three kernels:
//   1. blockTotals: every block sums its 2048-element chunk (fp64 total);
//   2. scanTotals:  one block turns the totals into exclusive offsets;
//   3. scanChunks:  every block scans its chunk in shared memory and adds its offset.
// Offsets are carried in fp64 so rounding does not accumulate across 50k chunks.
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kItemsPerThread = 8;
constexpr int kChunk = kBlockSize * kItemsPerThread;

__device__ __forceinline__ double warpInclusiveScan(double v) {
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const double other = __shfl_up_sync(0xffffffffu, v, offset);
        if (lane >= offset) v += other;
    }
    return v;
}

// Inclusive scan across the block; also returns the block total via *total.
__device__ double blockInclusiveScan(double v, double* total) {
    __shared__ double warp_totals[32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    v = warpInclusiveScan(v);
    if (lane == 31) warp_totals[warp] = v;
    __syncthreads();
    if (warp == 0) {
        double t = lane < blockDim.x / 32 ? warp_totals[lane] : 0.0;
        t = warpInclusiveScan(t);
        warp_totals[lane] = t;
    }
    __syncthreads();
    if (warp > 0) v += warp_totals[warp - 1];
    *total = warp_totals[blockDim.x / 32 - 1];
    __syncthreads();  // warp_totals is reused by the next call
    return v;
}

__global__ void blockTotals(const float* input, double* totals, int n) {
    const size_t base = static_cast<size_t>(blockIdx.x) * kChunk;
    double local = 0.0;
    for (int i = 0; i < kItemsPerThread; ++i) {
        const size_t g = base + i * kBlockSize + threadIdx.x;
        if (g < static_cast<size_t>(n)) local += input[g];
    }
    double total;
    blockInclusiveScan(local, &total);
    if (threadIdx.x == 0) totals[blockIdx.x] = total;
}

// Single block: exclusive scan of the chunk totals, 256 at a time with a carry.
__global__ void scanTotals(double* totals, int num_chunks) {
    double carry = 0.0;
    for (int start = 0; start < num_chunks; start += blockDim.x) {
        const int i = start + threadIdx.x;
        const double v = i < num_chunks ? totals[i] : 0.0;
        double sum;
        const double inclusive = blockInclusiveScan(v, &sum);
        if (i < num_chunks) totals[i] = carry + inclusive - v;
        carry += sum;
    }
}

__global__ void scanChunks(const float* input, float* output, const double* offsets, int n) {
    __shared__ float s_data[kChunk];
    const size_t base = static_cast<size_t>(blockIdx.x) * kChunk;
    // Coalesced load of the chunk.
    for (int i = threadIdx.x; i < kChunk; i += kBlockSize) {
        const size_t g = base + i;
        s_data[i] = g < static_cast<size_t>(n) ? input[g] : 0.0f;
    }
    __syncthreads();
    // Each thread scans its 8 consecutive items sequentially...
    float items[kItemsPerThread];
    float running = 0.0f;
    for (int i = 0; i < kItemsPerThread; ++i) {
        running += s_data[threadIdx.x * kItemsPerThread + i];
        items[i] = running;
    }
    // ...then the per-thread totals are scanned across the block.
    double total;
    const double thread_prefix = blockInclusiveScan(running, &total) - running + offsets[blockIdx.x];
    for (int i = 0; i < kItemsPerThread; ++i) {
        s_data[threadIdx.x * kItemsPerThread + i] = static_cast<float>(thread_prefix + items[i]);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kChunk; i += kBlockSize) {
        const size_t g = base + i;
        if (g < static_cast<size_t>(n)) output[g] = s_data[i];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    const int num_chunks = (N + kChunk - 1) / kChunk;
    double* totals = nullptr;
    cudaMalloc(&totals, num_chunks * sizeof(double));
    blockTotals<<<num_chunks, kBlockSize>>>(input, totals, N);
    scanTotals<<<1, kBlockSize>>>(totals, num_chunks);
    scanChunks<<<num_chunks, kBlockSize>>>(input, output, totals, N);
    cudaDeviceSynchronize();
    cudaFree(totals);
}

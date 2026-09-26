// Stream Compaction (LeetGPU)
// https://leetgpu.com/challenges/stream-compaction
//
// Stable compaction of the positive elements = exclusive scan of the 0/1
// predicate + scatter. Reduce-then-scan over 2048-element chunks:
//   1. chunkCounts: number of positives per chunk;
//   2. scanCounts:  one block turns counts into chunk offsets (+ total);
//   3. scatter:     per chunk, a block-wide scan of per-thread counts gives
//                   every element its output slot; the tail is zero-filled.
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kItemsPerThread = 8;
constexpr int kChunk = kBlockSize * kItemsPerThread;

__device__ int blockExclusiveScan(int v, int* total) {
    __shared__ int warp_totals[32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    int incl = v;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, incl, offset);
        if (lane >= offset) incl += other;
    }
    if (lane == 31) warp_totals[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        int t = lane < kBlockSize / 32 ? warp_totals[lane] : 0;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int other = __shfl_up_sync(0xffffffffu, t, offset);
            if (lane >= offset) t += other;
        }
        warp_totals[lane] = t;
    }
    __syncthreads();
    if (warp > 0) incl += warp_totals[warp - 1];
    *total = warp_totals[kBlockSize / 32 - 1];
    __syncthreads();
    return incl - v;
}

__global__ void chunkCounts(const float* a, int n, int* counts) {
    const size_t base = static_cast<size_t>(blockIdx.x) * kChunk;
    int c = 0;
    for (int i = 0; i < kItemsPerThread; ++i) {
        const size_t g = base + i * kBlockSize + threadIdx.x;
        c += (g < static_cast<size_t>(n) && a[g] > 0.0f);
    }
    int total;
    blockExclusiveScan(c, &total);
    if (threadIdx.x == 0) counts[blockIdx.x] = total;
}

__global__ void scanCounts(int* counts, int num_chunks, int* total_out) {
    int carry = 0;
    for (int start = 0; start < num_chunks; start += kBlockSize) {
        const int i = start + threadIdx.x;
        const int v = i < num_chunks ? counts[i] : 0;
        int total;
        const int excl = blockExclusiveScan(v, &total);
        if (i < num_chunks) counts[i] = carry + excl;
        carry += total;
    }
    if (threadIdx.x == 0) *total_out = carry;
}

__global__ void scatter(const float* a, int n, const int* offsets, float* out) {
    const size_t base = static_cast<size_t>(blockIdx.x) * kChunk + threadIdx.x * kItemsPerThread;
    int c = 0;
    for (int i = 0; i < kItemsPerThread; ++i) {
        const size_t g = base + i;
        c += (g < static_cast<size_t>(n) && a[g] > 0.0f);
    }
    int total;
    int pos = offsets[blockIdx.x] + blockExclusiveScan(c, &total);
    for (int i = 0; i < kItemsPerThread; ++i) {
        const size_t g = base + i;
        if (g < static_cast<size_t>(n)) {
            const float v = a[g];
            if (v > 0.0f) out[pos++] = v;
        }
    }
}

__global__ void zeroTail(float* out, int n, const int* total) {
    const int k = *total;
    for (int i = k + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] = 0.0f;
}

// A, out are device pointers
extern "C" void solve(const float* A, int N, float* out) {
    const int num_chunks = (N + kChunk - 1) / kChunk;
    int* counts = nullptr;
    cudaMalloc(&counts, (num_chunks + 1) * sizeof(int));
    chunkCounts<<<num_chunks, kBlockSize>>>(A, N, counts);
    scanCounts<<<1, kBlockSize>>>(counts, num_chunks, counts + num_chunks);
    scatter<<<num_chunks, kBlockSize>>>(A, N, counts, out);
    int blocks = (N + kBlockSize - 1) / kBlockSize;
    zeroTail<<<blocks > 4096 ? 4096 : blocks, kBlockSize>>>(out, N, counts + num_chunks);
    cudaDeviceSynchronize();
    cudaFree(counts);
}

// Segmented Prefix Sum (LeetGPU)
// https://leetgpu.com/challenges/segmented-prefix-sum
//
// Exclusive scan that restarts at every flag. A segmented scan is an ordinary
// scan over (flag, sum) pairs with the associative operator
//     (f1, s1) (+) (f2, s2) = (f1 | f2, f2 ? s2 : s1 + s2),
// so the reduce-then-scan structure of the plain prefix sum carries over:
//   1. chunkAggregates: (flag, sum-after-last-flag) per 2048-element chunk;
//   2. scanAggregates:  one block scans them into exclusive chunk carries;
//   3. scanChunks:      block-level segmented scan + carry, sequential per thread.
// Sums are fp64 (the reference also accumulates in fp64).
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kItemsPerThread = 8;
constexpr int kChunk = kBlockSize * kItemsPerThread;

struct Seg {
    int flag;
    double sum;
};

__device__ __forceinline__ Seg combine(Seg a, Seg b) { return Seg{a.flag | b.flag, b.flag ? b.sum : a.sum + b.sum}; }

// Inclusive segmented scan across the block; *total receives the block aggregate.
__device__ Seg blockScan(Seg v, Seg* total) {
    __shared__ int s_flag[32];
    __shared__ double s_sum[32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const Seg other{__shfl_up_sync(0xffffffffu, v.flag, offset), __shfl_up_sync(0xffffffffu, v.sum, offset)};
        if (lane >= offset) v = combine(other, v);
    }
    if (lane == 31) {
        s_flag[warp] = v.flag;
        s_sum[warp] = v.sum;
    }
    __syncthreads();
    if (warp == 0) {
        Seg t = lane < kBlockSize / 32 ? Seg{s_flag[lane], s_sum[lane]} : Seg{0, 0.0};
        for (int offset = 1; offset < 32; offset <<= 1) {
            const Seg other{__shfl_up_sync(0xffffffffu, t.flag, offset), __shfl_up_sync(0xffffffffu, t.sum, offset)};
            if (lane >= offset) t = combine(other, t);
        }
        s_flag[lane] = t.flag;
        s_sum[lane] = t.sum;
    }
    __syncthreads();
    if (warp > 0) v = combine(Seg{s_flag[warp - 1], s_sum[warp - 1]}, v);
    *total = Seg{s_flag[kBlockSize / 32 - 1], s_sum[kBlockSize / 32 - 1]};
    __syncthreads();
    return v;
}

__device__ Seg threadAggregate(const float* values, const int* flags, size_t base, int n) {
    Seg agg{0, 0.0};
    for (int i = 0; i < kItemsPerThread; ++i) {
        const size_t g = base + i;
        if (g < static_cast<size_t>(n)) agg = combine(agg, Seg{flags[g], values[g]});
    }
    return agg;
}

__global__ void chunkAggregates(const float* values, const int* flags, int* chunk_flag, double* chunk_sum, int n) {
    const size_t base = static_cast<size_t>(blockIdx.x) * kChunk + threadIdx.x * kItemsPerThread;
    Seg total;
    blockScan(threadAggregate(values, flags, base, n), &total);
    if (threadIdx.x == 0) {
        chunk_flag[blockIdx.x] = total.flag;
        chunk_sum[blockIdx.x] = total.sum;
    }
}

// Exclusive carries: carry[c] = aggregate of chunks [0, c) (its sum part).
__global__ void scanAggregates(const int* chunk_flag, double* chunk_sum, int num_chunks) {
    Seg carry{0, 0.0};
    for (int start = 0; start < num_chunks; start += kBlockSize) {
        const int i = start + threadIdx.x;
        const Seg v = i < num_chunks ? Seg{chunk_flag[i], chunk_sum[i]} : Seg{0, 0.0};
        Seg total;
        const Seg inclusive = blockScan(v, &total);
        // Exclusive = carry (+) inclusive-of-previous; recover via the previous lane's inclusive value.
        __shared__ int prev_flag[kBlockSize];
        __shared__ double prev_sum[kBlockSize];
        prev_flag[threadIdx.x] = inclusive.flag;
        prev_sum[threadIdx.x] = inclusive.sum;
        __syncthreads();
        const Seg before = threadIdx.x == 0 ? Seg{0, 0.0} : Seg{prev_flag[threadIdx.x - 1], prev_sum[threadIdx.x - 1]};
        const Seg exclusive = combine(carry, before);
        __syncthreads();
        if (i < num_chunks) chunk_sum[i] = exclusive.sum;
        carry = combine(carry, total);
    }
}

__global__ void scanChunks(const float* values, const int* flags, const double* chunk_carry, float* output, int n) {
    const size_t base = static_cast<size_t>(blockIdx.x) * kChunk + threadIdx.x * kItemsPerThread;
    const Seg agg = threadAggregate(values, flags, base, n);
    Seg total;
    const Seg inclusive = blockScan(agg, &total);
    // Exclusive thread prefix = inclusive minus own aggregate, rebuilt through shared memory.
    __shared__ int prev_flag[kBlockSize];
    __shared__ double prev_sum[kBlockSize];
    prev_flag[threadIdx.x] = inclusive.flag;
    prev_sum[threadIdx.x] = inclusive.sum;
    __syncthreads();
    const Seg before = threadIdx.x == 0 ? Seg{0, 0.0} : Seg{prev_flag[threadIdx.x - 1], prev_sum[threadIdx.x - 1]};
    double running = combine(Seg{0, chunk_carry[blockIdx.x]}, before).sum;
    for (int i = 0; i < kItemsPerThread; ++i) {
        const size_t g = base + i;
        if (g >= static_cast<size_t>(n)) break;
        if (flags[g]) running = 0.0;
        output[g] = static_cast<float>(running);
        running += values[g];
    }
}

// values, flags, output are device pointers
extern "C" void solve(const float* values, const int* flags, float* output, int N) {
    const int num_chunks = (N + kChunk - 1) / kChunk;
    void* buf = nullptr;
    cudaMalloc(&buf, num_chunks * (sizeof(double) + sizeof(int)));
    double* chunk_sum = static_cast<double*>(buf);
    int* chunk_flag = reinterpret_cast<int*>(chunk_sum + num_chunks);
    chunkAggregates<<<num_chunks, kBlockSize>>>(values, flags, chunk_flag, chunk_sum, N);
    scanAggregates<<<1, kBlockSize>>>(chunk_flag, chunk_sum, num_chunks);
    scanChunks<<<num_chunks, kBlockSize>>>(values, flags, chunk_sum, output, N);
    cudaDeviceSynchronize();
    cudaFree(buf);
}

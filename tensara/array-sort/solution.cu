// Array Sort (Tensara)
// https://tensara.org/problems/array-sort
//
// Ascending sort of int32 values: flipping the sign bit maps signed order onto
// unsigned order, then a stable LSD radix sort (4 x 8-bit passes) runs on the
// uint32 keys, and the sign bit is flipped back.
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Stable LSD radix sort of uint32 keys (4 passes of 8 bits).
// Per pass:
//   1. digitCounts: each 2048-key tile builds a 256-bin histogram (shared
//      atomics) and stores it digit-major: hist[digit * num_tiles + tile];
//   2. exclusiveScan over the whole table -> global start of (digit, tile);
//   3. scatterStable: each tile walks its keys in order, 256 at a time; within
//      a warp, __match_any_sync groups equal digits and popc(peers & lanes
//      below) ranks them; per-warp digit counts are prefix-summed across warps
//      so every key gets a unique, order-preserving slot.
// ---------------------------------------------------------------------------
constexpr int kSortThreads = 256;
constexpr int kSortWarps = kSortThreads / 32;
constexpr int kSortChunks = 8;
constexpr int kSortTile = kSortThreads * kSortChunks;
constexpr int kRadix = 256;
constexpr int kScanChunk = 2048;

__global__ void digitCounts(const unsigned int* keys, int n, int shift, unsigned int* hist, int num_tiles) {
    __shared__ unsigned int s_hist[kRadix];
    s_hist[threadIdx.x] = 0;
    __syncthreads();
    const size_t base = static_cast<size_t>(blockIdx.x) * kSortTile;
    for (int c = 0; c < kSortChunks; ++c) {
        const size_t i = base + c * kSortThreads + threadIdx.x;
        if (i < static_cast<size_t>(n)) atomicAdd(&s_hist[(keys[i] >> shift) & 0xFFu], 1u);
    }
    __syncthreads();
    hist[static_cast<size_t>(threadIdx.x) * num_tiles + blockIdx.x] = s_hist[threadIdx.x];
}

__device__ unsigned int blockExclusiveScanU(unsigned int v, unsigned int* total) {
    __shared__ unsigned int warp_totals[32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    unsigned int incl = v;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const unsigned int other = __shfl_up_sync(0xffffffffu, incl, offset);
        if (lane >= offset) incl += other;
    }
    if (lane == 31) warp_totals[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        unsigned int t = lane < static_cast<int>(blockDim.x / 32) ? warp_totals[lane] : 0u;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const unsigned int other = __shfl_up_sync(0xffffffffu, t, offset);
            if (lane >= offset) t += other;
        }
        warp_totals[lane] = t;
    }
    __syncthreads();
    if (warp > 0) incl += warp_totals[warp - 1];
    *total = warp_totals[blockDim.x / 32 - 1];
    __syncthreads();
    return incl - v;
}

// Device-wide exclusive scan in three kernels (chunk sums, scan of sums, apply).
__global__ void scanChunkSums(const unsigned int* data, int n, unsigned int* sums) {
    const size_t base = static_cast<size_t>(blockIdx.x) * kScanChunk;
    unsigned int local = 0;
    for (int i = threadIdx.x; i < kScanChunk; i += blockDim.x)
        if (base + i < static_cast<size_t>(n)) local += data[base + i];
    unsigned int total;
    blockExclusiveScanU(local, &total);
    if (threadIdx.x == 0) sums[blockIdx.x] = total;
}

__global__ void scanSums(unsigned int* sums, int count) {
    unsigned int carry = 0;
    for (int start = 0; start < count; start += blockDim.x) {
        const int i = start + threadIdx.x;
        const unsigned int v = i < count ? sums[i] : 0u;
        unsigned int total;
        const unsigned int excl = blockExclusiveScanU(v, &total);
        if (i < count) sums[i] = carry + excl;
        carry += total;
    }
}

// Each thread owns 8 consecutive entries of the chunk.
__global__ void scanApply(unsigned int* data, int n, const unsigned int* sums) {
    const size_t base = static_cast<size_t>(blockIdx.x) * kScanChunk + threadIdx.x * (kScanChunk / kSortThreads);
    unsigned int vals[kScanChunk / kSortThreads];
    unsigned int local = 0;
    for (int i = 0; i < kScanChunk / kSortThreads; ++i) {
        vals[i] = base + i < static_cast<size_t>(n) ? data[base + i] : 0u;
        local += vals[i];
    }
    unsigned int total;
    unsigned int run = sums[blockIdx.x] + blockExclusiveScanU(local, &total);
    for (int i = 0; i < kScanChunk / kSortThreads; ++i) {
        if (base + i < static_cast<size_t>(n)) data[base + i] = run;
        run += vals[i];
    }
}

static void exclusiveScan(unsigned int* data, int n, unsigned int* sums) {
    const int chunks = (n + kScanChunk - 1) / kScanChunk;
    scanChunkSums<<<chunks, kSortThreads>>>(data, n, sums);
    scanSums<<<1, kSortThreads>>>(sums, chunks);
    scanApply<<<chunks, kSortThreads>>>(data, n, sums);
}

__global__ void scatterStable(const unsigned int* keys_in, unsigned int* keys_out, int n, int shift,
                              const unsigned int* offsets, int num_tiles) {
    __shared__ unsigned int s_base[kRadix];
    __shared__ unsigned int s_warp[kSortWarps][kRadix];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    s_base[threadIdx.x] = offsets[static_cast<size_t>(threadIdx.x) * num_tiles + blockIdx.x];
    const size_t tile = static_cast<size_t>(blockIdx.x) * kSortTile;
    for (int c = 0; c < kSortChunks; ++c) {
        for (int w = 0; w < kSortWarps; ++w) s_warp[w][threadIdx.x] = 0;
        __syncthreads();
        const size_t i = tile + c * kSortThreads + threadIdx.x;
        const bool valid = i < static_cast<size_t>(n);
        const unsigned int key = valid ? keys_in[i] : 0u;
        const unsigned int digit = valid ? ((key >> shift) & 0xFFu) : kRadix;  // invalid keys form their own group
        const unsigned int peers = __match_any_sync(0xffffffffu, digit);
        const unsigned int rank = __popc(peers & ((1u << lane) - 1u));
        if (valid && lane == __ffs(peers) - 1) s_warp[warp][digit] = __popc(peers);
        __syncthreads();
        // Thread d turns the per-warp counts of digit d into start offsets.
        unsigned int run = s_base[threadIdx.x];
        for (int w = 0; w < kSortWarps; ++w) {
            const unsigned int cnt = s_warp[w][threadIdx.x];
            s_warp[w][threadIdx.x] = run;
            run += cnt;
        }
        s_base[threadIdx.x] = run;
        __syncthreads();
        if (valid) keys_out[s_warp[warp][digit] + rank] = key;
        __syncthreads();
    }
}

// Sorts keys[0..n) ascending; tmp has room for n keys. Result ends in keys.
static void radixSort(unsigned int* keys, unsigned int* tmp, int n) {
    const int num_tiles = (n + kSortTile - 1) / kSortTile;
    const int table = kRadix * num_tiles;
    unsigned int* hist = nullptr;
    cudaMalloc(&hist, (static_cast<size_t>(table) + (table + kScanChunk - 1) / kScanChunk) * sizeof(unsigned int));
    unsigned int* sums = hist + table;
    unsigned int* src = keys;
    unsigned int* dst = tmp;
    for (int shift = 0; shift < 32; shift += 8) {
        digitCounts<<<num_tiles, kSortThreads>>>(src, n, shift, hist, num_tiles);
        exclusiveScan(hist, table, sums);
        scatterStable<<<num_tiles, kSortThreads>>>(src, dst, n, shift, hist, num_tiles);
        unsigned int* t = src;
        src = dst;
        dst = t;
    }
    cudaFree(hist);  // 4 passes: the result is back in keys
}

__global__ void flipSign(const int* in, unsigned int* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<unsigned int>(in[i]) ^ 0x80000000u;
}

// a, b are device pointers
extern "C" void solution(const int* a, int* b, size_t n) {
    const int N = static_cast<int>(n);
    unsigned int* keys = reinterpret_cast<unsigned int*>(b);
    unsigned int* tmp = nullptr;
    cudaMalloc(&tmp, n * sizeof(unsigned int));
    flipSign<<<(N + 255) / 256, 256>>>(a, keys, N);
    radixSort(keys, tmp, N);
    flipSign<<<(N + 255) / 256, 256>>>(reinterpret_cast<const int*>(keys), keys, N);
    cudaDeviceSynchronize();
    cudaFree(tmp);
}

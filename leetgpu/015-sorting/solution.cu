// Sorting (LeetGPU)
// https://leetgpu.com/challenges/sorting
//
// Floats are mapped to order-preserving uint32 keys (negative: flip all bits;
// positive: set the sign bit), sorted with a stable LSD radix sort (4 x 8-bit
// passes, O(n) work), and mapped back in place.
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
    // Pass 1 of each digit: per-tile histogram of the current 8-bit digit (shared-memory atomics).
    __shared__ unsigned int s_hist[kRadix];
    s_hist[threadIdx.x] = 0;
    __syncthreads();
    const size_t base = static_cast<size_t>(blockIdx.x) * kSortTile;
    for (int c = 0; c < kSortChunks; ++c) {
        const size_t i = base + c * kSortThreads + threadIdx.x;
        if (i < static_cast<size_t>(n)) atomicAdd(&s_hist[(keys[i] >> shift) & 0xFFu], 1u);
    }
    __syncthreads();
    // Store digit-major (digit * num_tiles + tile): an exclusive scan of this table then gives
    // every (digit, tile) pair its global output offset, in stable order.
    hist[static_cast<size_t>(threadIdx.x) * num_tiles + blockIdx.x] = s_hist[threadIdx.x];
}

__device__ unsigned int blockExclusiveScanU(unsigned int v, unsigned int* total) {
    // Block-wide exclusive scan of unsigned values: warp scans, a scan of the warp totals, then combine.
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
    // Exclusive = inclusive - own value.
    return incl - v;
}

// Device-wide exclusive scan in three kernels (chunk sums, scan of sums, apply).
__global__ void scanChunkSums(const unsigned int* data, int n, unsigned int* sums) {
    // Scan pass 1: total of each 2048-element chunk.
    const size_t base = static_cast<size_t>(blockIdx.x) * kScanChunk;
    unsigned int local = 0;
    for (int i = threadIdx.x; i < kScanChunk; i += blockDim.x)
        if (base + i < static_cast<size_t>(n)) local += data[base + i];
    unsigned int total;
    blockExclusiveScanU(local, &total);
    if (threadIdx.x == 0) sums[blockIdx.x] = total;
}

__global__ void scanSums(unsigned int* sums, int count) {
    // Scan pass 2 (one block): exclusive scan of the chunk totals with a running carry.
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
    // Scan pass 3: each thread scans its 8 values serially from chunk carry + block prefix.
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
    // Three-kernel reduce-then-scan over the histogram table.
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
    // Pass 2 of each digit: stable scatter. Thread t owns digit t's running output position for this tile.
    s_base[threadIdx.x] = offsets[static_cast<size_t>(threadIdx.x) * num_tiles + blockIdx.x];
    const size_t tile = static_cast<size_t>(blockIdx.x) * kSortTile;
    // Process the tile 256 keys at a time, in order, so equal digits keep their input order.
    for (int c = 0; c < kSortChunks; ++c) {
        for (int w = 0; w < kSortWarps; ++w) s_warp[w][threadIdx.x] = 0;
        __syncthreads();
        const size_t i = tile + c * kSortThreads + threadIdx.x;
        const bool valid = i < static_cast<size_t>(n);
        const unsigned int key = valid ? keys_in[i] : 0u;
        const unsigned int digit = valid ? ((key >> shift) & 0xFFu) : kRadix;  // invalid keys form their own group
        // Lanes with the same digit find each other; rank = number of such lanes before me.
        // The first lane of each group records the group size for its warp.
        const unsigned int peers = __match_any_sync(0xffffffffu, digit);
        const unsigned int rank = __popc(peers & ((1u << lane) - 1u));
        if (valid && lane == __ffs(peers) - 1) s_warp[warp][digit] = __popc(peers);
        __syncthreads();
        // Thread d turns the per-warp counts of digit d into start offsets.
        // Thread t turns the per-warp counts of digit t into per-warp start offsets (warps in order).
        unsigned int run = s_base[threadIdx.x];
        for (int w = 0; w < kSortWarps; ++w) {
            const unsigned int cnt = s_warp[w][threadIdx.x];
            s_warp[w][threadIdx.x] = run;
            run += cnt;
        }
        s_base[threadIdx.x] = run;
        __syncthreads();
        // Scatter: warp offset for my digit + my rank inside the warp.
        if (valid) keys_out[s_warp[warp][digit] + rank] = key;
        __syncthreads();
    }
}

// Sorts keys[0..n) ascending; tmp has room for n keys. Result ends in keys.
static void radixSort(unsigned int* keys, unsigned int* tmp, int n) {
    // LSD radix sort: four 8-bit passes, ping-ponging between keys and tmp.
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

// Order-preserving float -> uint32 map (flip all bits of negatives, set the sign bit of positives)...
__global__ void floatToKey(const float* in, unsigned int* keys, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const unsigned int u = __float_as_uint(in[i]);
        keys[i] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }
}

// ...and its inverse.
__global__ void keyToFloat(const unsigned int* keys, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const unsigned int k = keys[i];
        out[i] = __uint_as_float((k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k);
    }
}

// data is a device pointer
extern "C" void solve(float* data, int N) {
    // Map to keys, LSD radix sort (keys + ping-pong buffer), map back in place.
    unsigned int* keys = nullptr;
    cudaMalloc(&keys, 2 * static_cast<size_t>(N) * sizeof(unsigned int));
    const int blocks = (N + 255) / 256;
    floatToKey<<<blocks, 256>>>(data, keys, N);
    radixSort(keys, keys + N, N);
    keyToFloat<<<blocks, 256>>>(keys, data, N);
    cudaDeviceSynchronize();
    cudaFree(keys);
}

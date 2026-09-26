// Chapter 11: scan (inclusive prefix sum) of int32 data.
//
//   warpInclusiveScan    Kogge-Stone scan across 32 lanes with __shfl_up_sync
//   blockInclusiveScan   warp scans + a scan of the warp totals
//   tile scan            8 items per thread: sequential scan in registers + block scan
//   scanReduceThenScan   three kernels: tile sums, scan of the sums, tile scans (reads 2n)
//   scanSinglePass       one kernel with decoupled look-back (reads n, like a copy)
//
// Integer data makes every result exact, so the checks compare bit for bit.
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 11-scan.cu -o scan
#include "check.cuh"

constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kThreads = 256;
constexpr int kItems = 8;                   // items per thread
constexpr int kTile = kThreads * kItems;    // 2048 items per block

// ---- Building blocks ---------------------------------------------------------------------
// After step d, lane l holds x[l-2d+1 .. l]: 5 steps for 32 lanes (Kogge-Stone / Hillis-Steele).
__device__ int warpInclusiveScan(int v) {
    const int lane = threadIdx.x % 32;
    for (int d = 1; d < 32; d <<= 1) {
        const int u = __shfl_up_sync(kFullMask, v, d);
        if (lane >= d) v += u;
    }
    return v;
}

// Inclusive scan of one value per thread across the block; *total receives the block sum.
__device__ int blockInclusiveScan(int v, int* total) {
    __shared__ int warp_totals[32];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int num_warps = blockDim.x / 32;
    v = warpInclusiveScan(v);
    if (lane == 31) warp_totals[warp] = v;          // each warp's total
    __syncthreads();
    if (warp == 0) {                                // warp 0 scans the (at most 32) totals
        const int t = lane < num_warps ? warp_totals[lane] : 0;
        warp_totals[lane] = warpInclusiveScan(t);
    }
    __syncthreads();
    if (warp > 0) v += warp_totals[warp - 1];       // add the totals of the warps before
    *total = warp_totals[num_warps - 1];
    __syncthreads();                                // warp_totals may be reused by the caller
    return v;
}

// Shared-memory index with one padding word per 32: thread t reading its kItems consecutive
// items (t * 8 + j) then hits 32 different banks for every j.
__device__ __forceinline__ int padded(int i) { return i + i / 32; }

// Scans one tile of kTile items starting at `offset`: coalesced load into shared memory,
// sequential scan of 8 items per thread in registers, block scan of the thread totals.
// Returns the tile total; the scanned tile (without any carry-in) is left in `tile`.
__device__ int scanTileInShared(const int* in, int* tile, int offset, int n) {
    for (int i = threadIdx.x; i < kTile; i += blockDim.x)   // coalesced: consecutive threads, consecutive items
        tile[padded(i)] = offset + i < n ? in[offset + i] : 0;
    __syncthreads();
    int items[kItems];
    int running = 0;
#pragma unroll
    for (int j = 0; j < kItems; ++j) {                       // sequential inclusive scan in registers
        running += tile[padded(threadIdx.x * kItems + j)];
        items[j] = running;
    }
    int total = 0;
    const int thread_inclusive = blockInclusiveScan(running, &total);
    const int carry = thread_inclusive - running;            // exclusive prefix of this thread
#pragma unroll
    for (int j = 0; j < kItems; ++j) tile[padded(threadIdx.x * kItems + j)] = items[j] + carry;
    __syncthreads();
    return total;
}

__device__ void storeTile(const int* tile, int* out, int offset, int n, int add) {
    for (int i = threadIdx.x; i < kTile; i += blockDim.x)
        if (offset + i < n) out[offset + i] = tile[padded(i)] + add;
}

// ---- Unit-test kernels for the building blocks -------------------------------------------
__global__ void warpScanKernel(const int* in, int* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = warpInclusiveScan(i < n ? in[i] : 0);
    if (i < n) out[i] = v;
}

__global__ void blockScanKernel(const int* in, int* out, int* totals, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 0;
    const int v = blockInclusiveScan(i < n ? in[i] : 0, &total);
    if (i < n) out[i] = v;
    if (threadIdx.x == 0) totals[blockIdx.x] = total;
}

// ---- Reduce-then-scan: three kernels ------------------------------------------------------
__global__ void tileSums(const int* in, int* sums, int n) {
    __shared__ int partial[kThreads / 32];
    const int offset = blockIdx.x * kTile;
    int v = 0;
    for (int i = threadIdx.x; i < kTile; i += blockDim.x)
        if (offset + i < n) v += in[offset + i];
    for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(kFullMask, v, d);
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        int s = 0;
        for (int w = 0; w < kThreads / 32; ++w) s += partial[w];
        sums[blockIdx.x] = s;
    }
}

// One block turns the tile sums into exclusive prefixes, kThreads at a time with a carry.
__global__ void scanSums(int* sums, int num_tiles) {
    int carry = 0;
    for (int base = 0; base < num_tiles; base += blockDim.x) {
        const int i = base + threadIdx.x;
        const int x = i < num_tiles ? sums[i] : 0;
        int total = 0;
        const int inclusive = blockInclusiveScan(x, &total);
        if (i < num_tiles) sums[i] = carry + inclusive - x;    // exclusive prefix
        carry += total;
    }
}

__global__ void scanTiles(const int* in, int* out, const int* tile_prefix, int n) {
    __shared__ int tile[kTile + kTile / 32];
    const int offset = blockIdx.x * kTile;
    scanTileInShared(in, tile, offset, n);
    storeTile(tile, out, offset, n, tile_prefix[blockIdx.x]);
}

void scanReduceThenScan(const int* in, int* out, int* sums, int n) {
    const int tiles = ex::ceilDiv(n, kTile);
    tileSums<<<tiles, kThreads>>>(in, sums, n);
    scanSums<<<1, 1024>>>(sums, tiles);
    scanTiles<<<tiles, kThreads>>>(in, out, sums, n);
}

// ---- Single pass with decoupled look-back -------------------------------------------------
// status[t] packs a flag (high 32 bits) and a value (low 32 bits) in one 64-bit word, so a
// reader always sees a flag together with its matching value.
constexpr unsigned kNotReady = 0;    // nothing published yet
constexpr unsigned kAggregate = 1;   // value = sum of tile t alone
constexpr unsigned kPrefix = 2;      // value = sum of tiles 0 .. t (inclusive prefix)

__device__ __forceinline__ unsigned long long packStatus(unsigned flag, int value) {
    return (static_cast<unsigned long long>(flag) << 32) | static_cast<unsigned>(value);
}

__global__ void scanSinglePass(const int* in, int* out, unsigned long long* status, int* tile_counter, int n) {
    __shared__ int tile[kTile + kTile / 32];
    __shared__ int s_tile, s_exclusive;
    // Tiles are numbered in the order blocks *start*, not by blockIdx: a tile's predecessors
    // have then always started, so waiting for them cannot deadlock.
    if (threadIdx.x == 0) s_tile = atomicAdd(tile_counter, 1);
    __syncthreads();
    const int t = s_tile;
    const int offset = t * kTile;
    const int total = scanTileInShared(in, tile, offset, n);

    if (threadIdx.x == 0) {
        volatile unsigned long long* vstatus = status;
        if (t == 0) {
            atomicExch(&status[0], packStatus(kPrefix, total));
            s_exclusive = 0;
        } else {
            atomicExch(&status[t], packStatus(kAggregate, total));   // let successors start early
            int prefix = 0;
            for (int p = t - 1; p >= 0; --p) {                       // look back
                unsigned long long s;
                do {
                    s = vstatus[p];
                } while ((s >> 32) == kNotReady);
                prefix += static_cast<int>(static_cast<unsigned>(s));
                if ((s >> 32) == kPrefix) break;                     // everything before p is included
            }
            atomicExch(&status[t], packStatus(kPrefix, prefix + total));
            s_exclusive = prefix;
        }
    }
    __syncthreads();
    storeTile(tile, out, offset, n, s_exclusive);
}

void scanSinglePassLaunch(const int* in, int* out, unsigned long long* status, int* counter, int n) {
    const int tiles = ex::ceilDiv(n, kTile);
    CUDA_CHECK(cudaMemsetAsync(status, 0, tiles * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemsetAsync(counter, 0, sizeof(int)));
    scanSinglePass<<<tiles, kThreads>>>(in, out, status, counter, n);
}

// ---- Checks -------------------------------------------------------------------------------
std::vector<int> randomInts(int n, uint32_t seed) {
    std::vector<int> v(n);
    for (int i = 0; i < n; ++i) v[i] = static_cast<int>(ex::randomValue(i, seed, -100.0f, 100.0f));
    return v;
}

std::vector<int> segmentedReference(const std::vector<int>& x, int segment) {
    std::vector<int> r(x.size());
    for (size_t i = 0; i < x.size(); ++i) r[i] = x[i] + (i % segment ? r[i - 1] : 0);
    return r;
}

void checkBuildingBlocks(int n) {
    const std::vector<int> x = randomInts(n, 1);
    ex::DeviceArray<int> d_in(x), d_out(n), d_totals(ex::ceilDiv(n, kThreads));
    warpScanKernel<<<ex::ceilDiv(n, kThreads), kThreads>>>(d_in.ptr, d_out.ptr, n);
    char name[64];
    std::snprintf(name, sizeof(name), "warpInclusiveScan n=%d", n);
    ex::check(name, d_out.download() == segmentedReference(x, 32));
    blockScanKernel<<<ex::ceilDiv(n, kThreads), kThreads>>>(d_in.ptr, d_out.ptr, d_totals.ptr, n);
    std::snprintf(name, sizeof(name), "blockInclusiveScan n=%d", n);
    ex::check(name, d_out.download() == segmentedReference(x, kThreads));
}

void checkDeviceScans(int n) {
    const std::vector<int> x = randomInts(n, 2);
    const std::vector<int> ref = segmentedReference(x, n);
    const int tiles = ex::ceilDiv(n, kTile);
    ex::DeviceArray<int> d_in(x), d_out(n), d_sums(tiles), d_counter(1);
    ex::DeviceArray<unsigned long long> d_status(tiles);
    char name[64];
    scanReduceThenScan(d_in.ptr, d_out.ptr, d_sums.ptr, n);
    std::snprintf(name, sizeof(name), "scanReduceThenScan n=%d", n);
    ex::check(name, d_out.download() == ref);
    CUDA_CHECK(cudaMemset(d_out.ptr, 0, n * sizeof(int)));
    scanSinglePassLaunch(d_in.ptr, d_out.ptr, d_status.ptr, d_counter.ptr, n);
    std::snprintf(name, sizeof(name), "scanSinglePass n=%d", n);
    ex::check(name, d_out.download() == ref);
}

void bench() {
    const int n = 1 << 26;
    const std::vector<int> x = randomInts(n, 3);
    const int tiles = ex::ceilDiv(n, kTile);
    ex::DeviceArray<int> d_in(x), d_out(n), d_sums(tiles), d_counter(1);
    ex::DeviceArray<unsigned long long> d_status(tiles);
    const double bytes = 8.0 * n;   // compulsory: read n ints, write n ints
    ex::reportBandwidth("cudaMemcpy (device to device)", ex::timeMs([&] {
        CUDA_CHECK(cudaMemcpyAsync(d_out.ptr, d_in.ptr, n * sizeof(int), cudaMemcpyDeviceToDevice));
    }), bytes);
    ex::reportBandwidth("scanReduceThenScan", ex::timeMs([&] {
        scanReduceThenScan(d_in.ptr, d_out.ptr, d_sums.ptr, n);
    }), bytes);
    ex::reportBandwidth("scanSinglePass (decoupled look-back)", ex::timeMs([&] {
        scanSinglePassLaunch(d_in.ptr, d_out.ptr, d_status.ptr, d_counter.ptr, n);
    }), bytes);
}

int main(int argc, char** argv) {
    for (int n : {1, 31, 257, 1000}) checkBuildingBlocks(n);
    for (int n : {1, 31, kTile, kTile + 1, 3 * kTile - 5, 100000}) checkDeviceScans(n);
    if (ex::wantBench(argc, argv)) bench();
    return ex::finish("11-scan");
}

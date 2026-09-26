// Chapter 10: warp-level primitives and cooperative groups.
//
//   1. warpAllReduceMax   butterfly all-reduce with __shfl_xor_sync
//   2. compactPositive    stream compaction: __ballot_sync + __popc, one atomic per warp
//   3. histogramMatch     __match_any_sync: lanes with the same key elect one leader
//   4. blockSumCg         the block reduction of chapter 03 written with cooperative groups
//   5. rowSums16          a 16-lane thread_block_tile per row: segmented reduction
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 10-warp-primitives.cu -o warp_primitives
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "check.cuh"

namespace cg = cooperative_groups;

constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kThreads = 256;

// ---- 1. All-reduce ---------------------------------------------------------------------
// After log2(32) = 5 butterfly steps every lane holds the maximum of the warp.
__device__ float warpAllReduceMax(float v) {
    for (int lane_mask = 16; lane_mask > 0; lane_mask >>= 1) v = fmaxf(v, __shfl_xor_sync(kFullMask, v, lane_mask));
    return v;
}

// out[i] = in[i] - max(in over the 32 elements of i's warp). Every lane needs the max,
// which is what the butterfly provides without a broadcast. Threads past the end still
// take part in the shuffles (with the identity -inf) and only skip the store.
__global__ void subtractWarpMax(const float* in, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float v = i < n ? in[i] : -INFINITY;
    const float m = warpAllReduceMax(v);
    if (i < n) out[i] = v - m;
}

// ---- 2. Stream compaction ----------------------------------------------------------------
// Keeps the positive elements. Each warp votes, reserves space for all its survivors with
// ONE atomicAdd, and every surviving lane computes its slot from the votes of the lanes
// before it. Within a warp the order is preserved; across warps it depends on the atomics.
__global__ void compactPositive(const float* in, float* out, int* count, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % 32;
    const bool keep = i < n && in[i] > 0.0f;
    const unsigned votes = __ballot_sync(kFullMask, keep);   // bit l: lane l keeps its element
    int base = 0;
    if (lane == 0 && votes != 0) base = atomicAdd(count, __popc(votes));
    base = __shfl_sync(kFullMask, base, 0);                    // broadcast the warp's base
    const unsigned lanes_before = votes & ((1u << lane) - 1u);  // survivors in lanes < mine
    if (keep) out[base + __popc(lanes_before)] = in[i];
}

// The baseline: one atomic per surviving element (32x more atomics on the same address).
__global__ void compactPositiveNaive(const float* in, float* out, int* count, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && in[i] > 0.0f) out[atomicAdd(count, 1)] = in[i];
}

// ---- 3. Histogram with __match_any_sync -------------------------------------------------
constexpr int kBins = 64;

// Per-block histogram in shared memory. Lanes of a warp that hold the same key find each
// other with __match_any_sync; only the lowest of them adds the whole group's count.
// With skewed data (many equal keys) this removes most shared-memory atomic conflicts.
__global__ void histogramMatch(const int* keys, int* hist, int n) {
    __shared__ int local[kBins];
    for (int b = threadIdx.x; b < kBins; b += blockDim.x) local[b] = 0;
    __syncthreads();
    const int lane = threadIdx.x % 32;
    // The loop bound is the same for the whole block, so every lane reaches the *_sync call.
    for (int base = blockIdx.x * blockDim.x; base < n; base += gridDim.x * blockDim.x) {
        const int i = base + threadIdx.x;
        const int key = i < n ? keys[i] : -1;                    // -1: "no element"
        const unsigned peers = __match_any_sync(kFullMask, key);  // lanes holding the same key
        const int leader = __ffs(peers) - 1;                      // lowest lane of the group
        if (key >= 0 && lane == leader) atomicAdd(&local[key], __popc(peers));
    }
    __syncthreads();
    for (int b = threadIdx.x; b < kBins; b += blockDim.x)
        if (local[b] != 0) atomicAdd(&hist[b], local[b]);
}

// ---- 4. Block reduction with cooperative groups -----------------------------------------
__global__ void blockSumCg(const float* in, float* block_sums, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float warp_sums[32];

    float v = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) v += in[i];
    v = cg::reduce(warp, v, cg::plus<float>());                  // every lane gets the warp sum
    if (warp.thread_rank() == 0) warp_sums[warp.meta_group_rank()] = v;
    block.sync();
    if (warp.meta_group_rank() == 0) {
        v = warp.thread_rank() < warp.meta_group_size() ? warp_sums[warp.thread_rank()] : 0.0f;
        v = cg::reduce(warp, v, cg::plus<float>());
        if (warp.thread_rank() == 0) block_sums[blockIdx.x] = v;
    }
}

// ---- 5. Segmented reduction with a 16-lane tile -----------------------------------------
// in is rows x 16; each 16-lane tile reduces one row. Two rows per warp, no shared memory.
__global__ void rowSums16(const float* in, float* out, int rows) {
    const cg::thread_block_tile<16> tile = cg::tiled_partition<16>(cg::this_thread_block());
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 16;
    float v = row < rows ? in[static_cast<size_t>(row) * 16 + tile.thread_rank()] : 0.0f;
    v = cg::reduce(tile, v, cg::plus<float>());
    if (row < rows && tile.thread_rank() == 0) out[row] = v;
}

// ---- Checks -----------------------------------------------------------------------------
void checkWarpMax(int n) {
    const std::vector<float> x = ex::randomVector(n, 1);
    ex::DeviceArray<float> d_in(x), d_out(n);
    subtractWarpMax<<<ex::ceilDiv(n, kThreads), kThreads>>>(d_in.ptr, d_out.ptr, n);
    std::vector<double> ref(n);
    for (int w = 0; w < n; w += 32) {
        float m = -INFINITY;
        for (int i = w; i < std::min(n, w + 32); ++i) m = std::max(m, x[i]);
        for (int i = w; i < std::min(n, w + 32); ++i) ref[i] = x[i] - m;
    }
    char name[64];
    std::snprintf(name, sizeof(name), "subtractWarpMax n=%d", n);
    ex::checkClose(name, d_out.download(), ref, 0.0, 1e-6);
}

template <class Kernel>
void checkCompaction(const char* label, Kernel kernel, int n) {
    const std::vector<float> x = ex::randomVector(n, 2);
    ex::DeviceArray<float> d_in(x), d_out(n);
    ex::DeviceArray<int> d_count(1);
    d_count.zero();
    kernel<<<ex::ceilDiv(n, kThreads), kThreads>>>(d_in.ptr, d_out.ptr, d_count.ptr, n);
    const int count = d_count.download()[0];
    std::vector<float> got = d_out.download();
    got.resize(count);
    std::vector<float> ref;
    for (float v : x)
        if (v > 0.0f) ref.push_back(v);
    std::sort(got.begin(), got.end());   // the order across warps is not defined
    std::sort(ref.begin(), ref.end());
    char name[64];
    std::snprintf(name, sizeof(name), "%s n=%d (kept %d)", label, n, count);
    ex::check(name, got == ref);
}

void checkHistogram(int n) {
    std::vector<int> keys(n);
    for (int i = 0; i < n; ++i) {   // skewed: half of the keys are 7
        const float r = ex::randomValue(i, 3, 0.0f, 1.0f);
        keys[i] = r < 0.5f ? 7 : static_cast<int>(r * 2 * kBins) % kBins;
    }
    ex::DeviceArray<int> d_keys(keys), d_hist(kBins);
    d_hist.zero();
    histogramMatch<<<8, kThreads>>>(d_keys.ptr, d_hist.ptr, n);
    std::vector<int> ref(kBins, 0);
    for (int k : keys) ++ref[k];
    char name[64];
    std::snprintf(name, sizeof(name), "histogramMatch n=%d", n);
    ex::check(name, d_hist.download() == ref);
}

void checkBlockSum(int n, int blocks) {
    const std::vector<float> x = ex::randomVector(n, 4);
    ex::DeviceArray<float> d_in(x), d_sums(blocks);
    blockSumCg<<<blocks, kThreads>>>(d_in.ptr, d_sums.ptr, n);
    const std::vector<float> sums = d_sums.download();
    double got = 0.0, ref = 0.0, mag = 0.0;
    for (float s : sums) got += s;
    for (float v : x) {
        ref += v;
        mag += std::fabs(v);
    }
    char name[64];
    std::snprintf(name, sizeof(name), "blockSumCg n=%d blocks=%d", n, blocks);
    ex::check(name, std::fabs(got - ref) <= 1e-5 * mag + 1e-6);
}

void checkRowSums(int rows) {
    const std::vector<float> x = ex::randomVector(static_cast<size_t>(rows) * 16, 5);
    ex::DeviceArray<float> d_in(x), d_out(rows);
    rowSums16<<<ex::ceilDiv(rows * 16, kThreads), kThreads>>>(d_in.ptr, d_out.ptr, rows);
    std::vector<double> ref(rows, 0.0);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < 16; ++c) ref[r] += x[r * 16 + c];
    char name[64];
    std::snprintf(name, sizeof(name), "rowSums16 rows=%d", rows);
    ex::checkClose(name, d_out.download(), ref, 1e-5, 1e-5);
}

void bench() {
    const int n = 1 << 26;
    const std::vector<float> x = ex::randomVector(n, 6);
    ex::DeviceArray<float> d_in(x), d_out(n);
    ex::DeviceArray<int> d_count(1);
    const int blocks = ex::ceilDiv(n, kThreads);
    const double bytes = 4.0 * n * 1.5;   // read all, write about half
    ex::reportBandwidth("compactPositiveNaive", ex::timeMs([&] {
        d_count.zero();
        compactPositiveNaive<<<blocks, kThreads>>>(d_in.ptr, d_out.ptr, d_count.ptr, n);
    }), bytes);
    ex::reportBandwidth("compactPositive (warp-aggregated)", ex::timeMs([&] {
        d_count.zero();
        compactPositive<<<blocks, kThreads>>>(d_in.ptr, d_out.ptr, d_count.ptr, n);
    }), bytes);
}

int main(int argc, char** argv) {
    for (int n : {1, 31, 32, 1000, 4099}) checkWarpMax(n);
    for (int n : {1, 100, 5000}) checkCompaction("compactPositive", compactPositive, n);
    checkCompaction("compactPositiveNaive", compactPositiveNaive, 5000);
    for (int n : {1, 300, 10000}) checkHistogram(n);
    checkBlockSum(1, 1);
    checkBlockSum(100000, 7);
    for (int rows : {1, 3, 257}) checkRowSums(rows);
    if (ex::wantBench(argc, argv)) bench();
    return ex::finish("10-warp-primitives");
}

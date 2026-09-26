// Running Sum 1D (Tensara)
// https://tensara.org/problems/running-sum-1d
//
// out[i] = sum_{j < W} x[i + j - W/2] with zero padding (conv1d with a kernel
// of ones and padding W/2; the output has N + 2 (W/2) - W + 1 elements).
// With an inclusive prefix sum P (fp64), every window is P[hi] - P[lo]: O(N)
// total instead of O(N * W). The prefix sums are kept in double because the
// windows subtract two large, nearly equal numbers.
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Inclusive scan of floats with an associative operator Op (fp64 carries):
//   1. chunkTotals: each 2048-element chunk folds to one value;
//   2. scanTotals:  one block turns them into exclusive carries;
//   3. scanChunks:  block-level scan of each chunk (8 items per thread, then a
//                   warp-shuffle scan of the per-thread totals) plus the carry.
// ---------------------------------------------------------------------------
constexpr int kScanThreads = 256;
constexpr int kItems = 8;
constexpr int kChunk = kScanThreads * kItems;

template <class Op>
__device__ double blockInclusiveScan(double v, double* total) {
    // Block-wide inclusive scan of one value per thread; also returns the block total.
    __shared__ double warp_totals[32];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    // 1) Hillis-Steele scan inside each warp: lane l adds the value of lane l - offset.
    for (int offset = 1; offset < 32; offset <<= 1) {
        const double other = __shfl_up_sync(0xffffffffu, v, offset);
        if (lane >= offset) v = Op::apply(other, v);
    }
    // 2) The last lane of each warp publishes the warp total.
    if (lane == 31) warp_totals[warp] = v;
    __syncthreads();
    // 3) Warp 0 scans the warp totals (unused slots hold the identity).
    if (warp == 0) {
        double t = lane < kScanThreads / 32 ? warp_totals[lane] : Op::identity();
        for (int offset = 1; offset < 32; offset <<= 1) {
            const double other = __shfl_up_sync(0xffffffffu, t, offset);
            if (lane >= offset) t = Op::apply(other, t);
        }
        warp_totals[lane] = t;
    }
    __syncthreads();
    // 4) Add the inclusive total of all previous warps.
    if (warp > 0) v = Op::apply(warp_totals[warp - 1], v);
    *total = warp_totals[kScanThreads / 32 - 1];
    // Keep warp_totals intact until every thread has read it (the helper is called in loops).
    __syncthreads();
    return v;
}

template <class Op>
__global__ void chunkTotals(const float* in, double* totals, size_t n) {
    // Pass 1: each block folds its 2048-element chunk (8 consecutive items per thread) to one total.
    const size_t base = static_cast<size_t>(blockIdx.x) * kChunk + threadIdx.x * kItems;
    double local = Op::identity();
    for (int i = 0; i < kItems; ++i)
        if (base + i < n) local = Op::apply(local, in[base + i]);
    double total;
    blockInclusiveScan<Op>(local, &total);
    if (threadIdx.x == 0) totals[blockIdx.x] = total;
}

template <class Op>
__global__ void scanTotals(double* totals, int count) {
    // Pass 2 (one block): turn the chunk totals into exclusive carries, 256 at a time,
    // carrying the running total from one pass to the next.
    double carry = Op::identity();
    for (int start = 0; start < count; start += kScanThreads) {
        const int i = start + threadIdx.x;
        const double v = i < count ? totals[i] : Op::identity();
        double total;
        const double incl = blockInclusiveScan<Op>(v, &total);
        __shared__ double s_incl[kScanThreads];
        s_incl[threadIdx.x] = incl;
        __syncthreads();
        // Exclusive value = inclusive value of the previous thread.
        const double excl = threadIdx.x == 0 ? Op::identity() : s_incl[threadIdx.x - 1];
        if (i < count) totals[i] = Op::apply(carry, excl);
        __syncthreads();
        carry = Op::apply(carry, total);
    }
}

template <class Op>
__global__ void scanChunks(const float* in, double* out, const double* carries, size_t n) {
    const size_t base = static_cast<size_t>(blockIdx.x) * kChunk + threadIdx.x * kItems;
    // Pass 3: reload the chunk and fold each thread's 8 items...
    double items[kItems];
    double local = Op::identity();
    for (int i = 0; i < kItems; ++i) {
        items[i] = base + i < n ? static_cast<double>(in[base + i]) : Op::identity();
        local = Op::apply(local, items[i]);
    }
    double total;
    // ...scan the per-thread totals across the block...
    const double incl = blockInclusiveScan<Op>(local, &total);
    __shared__ double s_incl[kScanThreads];
    s_incl[threadIdx.x] = incl;
    __syncthreads();
    // ...and walk the 8 items serially from chunk carry + this thread's exclusive prefix.
    // The prefix sums stay in double (the window sums subtract two large, nearly equal values).
    double run = Op::apply(carries[blockIdx.x], threadIdx.x == 0 ? Op::identity() : s_incl[threadIdx.x - 1]);
    for (int i = 0; i < kItems; ++i) {
        run = Op::apply(run, items[i]);
        if (base + i < n) out[base + i] = run;
    }
}

template <class Op>
static void inclusiveScan(const float* in, double* out, size_t n) {
    // Reduce-then-scan: chunk totals, one-block scan of the totals, then per-chunk scans.
    const int chunks = static_cast<int>((n + kChunk - 1) / kChunk);
    double* totals = nullptr;
    cudaMalloc(&totals, chunks * sizeof(double));
    chunkTotals<Op><<<chunks, kScanThreads>>>(in, totals, n);
    scanTotals<Op><<<1, kScanThreads>>>(totals, chunks);
    scanChunks<Op><<<chunks, kScanThreads>>>(in, out, totals, n);
    cudaDeviceSynchronize();
    cudaFree(totals);
}

// The scan operator: addition in fp64, identity 0.
struct Plus {
    __device__ static double identity() { return 0.0; }
    __device__ static double apply(double a, double b) { return a + b; }
};

// prefix[i] = x[0] + ... + x[i]; window [lo, hi] is prefix[hi] - prefix[lo - 1].
__global__ void windowSums(const double* prefix, float* out, long long n, long long w, long long out_len) {
    // One thread per output: window i covers inputs [i - half, i - half + w - 1],
    // clipped to [0, n - 1] (zero padding contributes nothing).
    const long long half = w / 2;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < out_len; i += static_cast<long long>(gridDim.x) * blockDim.x) {
        long long lo = i - half;
        long long hi = i - half + w - 1;
        lo = lo < 0 ? 0 : lo;
        hi = hi > n - 1 ? n - 1 : hi;
        // Window sum = P[hi] - P[lo - 1] from the inclusive prefix sums, rounded to float once.
        double s = 0.0;
        if (hi >= lo) s = prefix[hi] - (lo > 0 ? prefix[lo - 1] : 0.0);
        out[i] = static_cast<float>(s);
    }
}

// input, output are device pointers
extern "C" void solution(const float* input, size_t W, float* output, size_t N) {
    // Inclusive fp64 prefix sum of the whole signal into a temporary buffer.
    double* prefix = nullptr;
    cudaMalloc(&prefix, N * sizeof(double));
    inclusiveScan<Plus>(input, prefix, N);
    // conv1d output length N + 2 (W/2) - W + 1 (equal to N for odd W).
    const long long out_len = static_cast<long long>(N) + 2 * static_cast<long long>(W / 2) - static_cast<long long>(W) + 1;
    long long blocks = (out_len + 255) / 256;
    blocks = blocks > 4096 ? 4096 : (blocks < 1 ? 1 : blocks);
    windowSums<<<static_cast<unsigned>(blocks), 256>>>(prefix, output, static_cast<long long>(N), static_cast<long long>(W), out_len);
    // Wait for the kernels before freeing the temporary buffer.
    cudaDeviceSynchronize();
    cudaFree(prefix);
}

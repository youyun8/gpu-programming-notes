// Cumulative Product (Tensara)
// https://tensara.org/problems/cumprod
//
// Inclusive scan with operator (a * b), reduce-then-scan over 2048-element chunks
// with fp64 carries so rounding does not build up across chunks.
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
    __shared__ double warp_totals[32];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const double other = __shfl_up_sync(0xffffffffu, v, offset);
        if (lane >= offset) v = Op::apply(other, v);
    }
    if (lane == 31) warp_totals[warp] = v;
    __syncthreads();
    if (warp == 0) {
        double t = lane < kScanThreads / 32 ? warp_totals[lane] : Op::identity();
        for (int offset = 1; offset < 32; offset <<= 1) {
            const double other = __shfl_up_sync(0xffffffffu, t, offset);
            if (lane >= offset) t = Op::apply(other, t);
        }
        warp_totals[lane] = t;
    }
    __syncthreads();
    if (warp > 0) v = Op::apply(warp_totals[warp - 1], v);
    *total = warp_totals[kScanThreads / 32 - 1];
    __syncthreads();
    return v;
}

template <class Op>
__global__ void chunkTotals(const float* in, double* totals, size_t n) {
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
    double carry = Op::identity();
    for (int start = 0; start < count; start += kScanThreads) {
        const int i = start + threadIdx.x;
        const double v = i < count ? totals[i] : Op::identity();
        double total;
        const double incl = blockInclusiveScan<Op>(v, &total);
        __shared__ double s_incl[kScanThreads];
        s_incl[threadIdx.x] = incl;
        __syncthreads();
        const double excl = threadIdx.x == 0 ? Op::identity() : s_incl[threadIdx.x - 1];
        if (i < count) totals[i] = Op::apply(carry, excl);
        __syncthreads();
        carry = Op::apply(carry, total);
    }
}

template <class Op>
__global__ void scanChunks(const float* in, float* out, const double* carries, size_t n) {
    const size_t base = static_cast<size_t>(blockIdx.x) * kChunk + threadIdx.x * kItems;
    double items[kItems];
    double local = Op::identity();
    for (int i = 0; i < kItems; ++i) {
        items[i] = base + i < n ? static_cast<double>(in[base + i]) : Op::identity();
        local = Op::apply(local, items[i]);
    }
    double total;
    const double incl = blockInclusiveScan<Op>(local, &total);
    __shared__ double s_incl[kScanThreads];
    s_incl[threadIdx.x] = incl;
    __syncthreads();
    double run = Op::apply(carries[blockIdx.x], threadIdx.x == 0 ? Op::identity() : s_incl[threadIdx.x - 1]);
    for (int i = 0; i < kItems; ++i) {
        run = Op::apply(run, items[i]);
        if (base + i < n) out[base + i] = static_cast<float>(run);
    }
}

template <class Op>
static void inclusiveScan(const float* in, float* out, size_t n) {
    const int chunks = static_cast<int>((n + kChunk - 1) / kChunk);
    double* totals = nullptr;
    cudaMalloc(&totals, chunks * sizeof(double));
    chunkTotals<Op><<<chunks, kScanThreads>>>(in, totals, n);
    scanTotals<Op><<<1, kScanThreads>>>(totals, chunks);
    scanChunks<Op><<<chunks, kScanThreads>>>(in, out, totals, n);
    cudaDeviceSynchronize();
    cudaFree(totals);
}

struct Times {
    __device__ static double identity() { return 1.0; }
    __device__ static double apply(double a, double b) { return a * b; }
};

// input, output are device pointers
extern "C" void solution(const float* input, float* output, size_t N) {
    inclusiveScan<Times>(input, output, N);
}

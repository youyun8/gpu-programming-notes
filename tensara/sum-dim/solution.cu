// Sum over Dimension (Tensara)
// https://tensara.org/problems/sum-dim
//
// Sum along `dim` (keepdim).
// The tensor is viewed as (outer, R, inner) with R = shape[dim]; the output
// has outer * inner elements.
//   - inner == 1 (reducing the contiguous axis): one warp per output, lanes
//     stride the row (coalesced), shuffle reduction;
//   - inner > 1: one thread per output, looping over R with stride inner, so
//     neighbouring threads read neighbouring addresses (coalesced).
// `shape` may be a host or a device pointer depending on the harness, so it is
// read with cudaMemcpyDefault (unified addressing resolves the direction).
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

constexpr int kBlockSize = 256;

// Reduction state: running sum.
struct Acc {
    float v;
    __device__ static Acc identity() { return Acc{0.0f}; }
    __device__ static Acc make(float x, int) { return Acc{x}; }
    __device__ static Acc combine(Acc a, Acc b) { return Acc{a.v + b.v}; }
    __device__ static Acc shuffle(Acc a, int offset) { return Acc{__shfl_down_sync(0xffffffffu, a.v, offset)}; }
    __device__ float finish(int r) const { return v; }
};

__global__ void reduceContiguous(const float* __restrict__ in, float* __restrict__ out, long long outer, int r) {
    const int lane = threadIdx.x % 32;
    const long long row = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (row >= outer) return;
    const float* p = in + row * r;
    Acc acc = Acc::identity();
    for (int j = lane; j < r; j += 32) acc = Acc::combine(acc, Acc::make(p[j], j));
    for (int offset = 16; offset > 0; offset >>= 1) acc = Acc::combine(acc, Acc::shuffle(acc, offset));
    if (lane == 0) out[row] = acc.finish(r);
}

__global__ void reduceStrided(const float* __restrict__ in, float* __restrict__ out, long long outer, int r, long long inner) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;
    const long long o = idx / inner;
    const long long i = idx % inner;
    const float* p = in + o * r * inner + i;
    Acc acc = Acc::identity();
    for (int j = 0; j < r; ++j) acc = Acc::combine(acc, Acc::make(p[static_cast<long long>(j) * inner], j));
    out[idx] = acc.finish(r);
}

// input, output are device pointers
extern "C" void solution(const float* input, int dim, float* output, const size_t* shape, size_t ndim) {
    size_t host_shape[16];
    cudaMemcpy(host_shape, shape, ndim * sizeof(size_t), cudaMemcpyDefault);
    long long outer = 1, inner = 1;
    for (int d = 0; d < static_cast<int>(ndim); ++d) {
        if (d < dim) outer *= static_cast<long long>(host_shape[d]);
        if (d > dim) inner *= static_cast<long long>(host_shape[d]);
    }
    const int r = static_cast<int>(host_shape[dim]);
    if (inner == 1) {
        const long long threads = outer * 32;
        reduceContiguous<<<static_cast<unsigned>((threads + kBlockSize - 1) / kBlockSize), kBlockSize>>>(input, output, outer, r);
    } else {
        const long long total = outer * inner;
        reduceStrided<<<static_cast<unsigned>((total + kBlockSize - 1) / kBlockSize), kBlockSize>>>(input, output, outer, r, inner);
    }
    cudaDeviceSynchronize();
}

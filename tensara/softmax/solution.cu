// Softmax over Dimension (Tensara)
// https://tensara.org/problems/softmax
//
// softmax along `dim` of an arbitrary-rank tensor, viewed as (outer, R, inner).
//   - inner == 1: one warp per row; each lane keeps an online (max, sum) pair
//     over its strided elements, pairs are merged with shuffles, then the
//     row is normalized - two reads + one write;
//   - inner > 1: one thread per (outer, inner) column, the same online pass
//     over R with stride inner (coalesced across threads).
// `shape` may be a host or device pointer: read it with cudaMemcpyDefault.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kBlockSize = 256;

// Online-softmax merge of two (max, sum of exp(x - max)) pairs: rescale both sums to the new max.
__device__ __forceinline__ void merge(float& m, float& s, float om, float os) {
    const float nm = fmaxf(m, om);
    s = s * expf(m - nm) + os * expf(om - nm);
    m = nm;
}

__global__ void softmaxRows(const float* in, float* out, long long rows, int r) {
    // Contiguous case: one warp per row.
    const int lane = threadIdx.x % 32;
    const long long row = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (row >= rows) return;
    const float* p = in + row * r;
    float* q = out + row * r;
    // Pass 1: each lane folds its strided elements into (m, s) in one sweep...
    float m = -FLT_MAX, s = 0.0f;
    for (int j = lane; j < r; j += 32) merge(m, s, p[j], 1.0f);
    // ...and a butterfly merges the 32 pairs, so every lane holds the row's max and sum.
    for (int o = 16; o > 0; o >>= 1) merge(m, s, __shfl_xor_sync(0xffffffffu, m, o), __shfl_xor_sync(0xffffffffu, s, o));
    // Pass 2: write exp(x - max) / sum.
    const float inv = 1.0f / s;
    for (int j = lane; j < r; j += 32) q[j] = expf(p[j] - m) * inv;
}

__global__ void softmaxStrided(const float* in, float* out, long long outer, int r, long long inner) {
    // Strided case: one thread per (outer, inner) column; neighbouring threads read
    // neighbouring addresses at every step of the walk along the reduced axis.
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;
    const long long base = (idx / inner) * r * inner + idx % inner;
    // Online (max, sum) over the column, then a second walk writes the probabilities.
    float m = -FLT_MAX, s = 0.0f;
    for (int j = 0; j < r; ++j) merge(m, s, in[base + static_cast<long long>(j) * inner], 1.0f);
    const float inv = 1.0f / s;
    for (int j = 0; j < r; ++j) out[base + static_cast<long long>(j) * inner] = expf(in[base + static_cast<long long>(j) * inner] - m) * inv;
}

// input, output are device pointers
extern "C" void solution(const float* input, int dim, float* output, const size_t* shape, size_t ndim) {
    // `shape` may be a host or a device pointer: cudaMemcpyDefault handles both.
    size_t host_shape[16];
    cudaMemcpy(host_shape, shape, ndim * sizeof(size_t), cudaMemcpyDefault);
    // View the tensor as (outer, r, inner) around the softmax dimension.
    long long outer = 1, inner = 1;
    for (int d = 0; d < static_cast<int>(ndim); ++d) {
        if (d < dim) outer *= static_cast<long long>(host_shape[d]);
        if (d > dim) inner *= static_cast<long long>(host_shape[d]);
    }
    const int r = static_cast<int>(host_shape[dim]);
    // Warp-per-row kernel for the contiguous last axis, thread-per-column otherwise.
    if (inner == 1) {
        softmaxRows<<<static_cast<unsigned>((outer * 32 + kBlockSize - 1) / kBlockSize), kBlockSize>>>(input, output, outer, r);
    } else {
        softmaxStrided<<<static_cast<unsigned>((outer * inner + kBlockSize - 1) / kBlockSize), kBlockSize>>>(input, output, outer, r, inner);
    }
    cudaDeviceSynchronize();
}

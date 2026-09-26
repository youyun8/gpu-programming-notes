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

__device__ __forceinline__ void merge(float& m, float& s, float om, float os) {
    const float nm = fmaxf(m, om);
    s = s * expf(m - nm) + os * expf(om - nm);
    m = nm;
}

__global__ void softmaxRows(const float* in, float* out, long long rows, int r) {
    const int lane = threadIdx.x % 32;
    const long long row = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (row >= rows) return;
    const float* p = in + row * r;
    float* q = out + row * r;
    float m = -FLT_MAX, s = 0.0f;
    for (int j = lane; j < r; j += 32) merge(m, s, p[j], 1.0f);
    for (int o = 16; o > 0; o >>= 1) merge(m, s, __shfl_xor_sync(0xffffffffu, m, o), __shfl_xor_sync(0xffffffffu, s, o));
    const float inv = 1.0f / s;
    for (int j = lane; j < r; j += 32) q[j] = expf(p[j] - m) * inv;
}

__global__ void softmaxStrided(const float* in, float* out, long long outer, int r, long long inner) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;
    const long long base = (idx / inner) * r * inner + idx % inner;
    float m = -FLT_MAX, s = 0.0f;
    for (int j = 0; j < r; ++j) merge(m, s, in[base + static_cast<long long>(j) * inner], 1.0f);
    const float inv = 1.0f / s;
    for (int j = 0; j < r; ++j) out[base + static_cast<long long>(j) * inner] = expf(in[base + static_cast<long long>(j) * inner] - m) * inv;
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
        softmaxRows<<<static_cast<unsigned>((outer * 32 + kBlockSize - 1) / kBlockSize), kBlockSize>>>(input, output, outer, r);
    } else {
        softmaxStrided<<<static_cast<unsigned>((outer * inner + kBlockSize - 1) / kBlockSize), kBlockSize>>>(input, output, outer, r, inner);
    }
    cudaDeviceSynchronize();
}

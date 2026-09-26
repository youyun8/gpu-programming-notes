// Log Softmax (Tensara)
// https://tensara.org/problems/log-softmax
//
// log_softmax over the columns of an M x N matrix: x - (max + log(sum e^(x - max))).
// One warp per row: online (max, sum) per lane, shuffle merge, then one
// write pass - the row is read twice and written once.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kBlockSize = 256;

// Online-softmax merge of two (max, sum of exp(x - max)) pairs: rescale both sums to the new max.
__device__ __forceinline__ void merge(float& m, float& s, float om, float os) {
    const float nm = fmaxf(m, om);
    s = s * expf(m - nm) + os * expf(om - nm);
    m = nm;
}

__global__ void logSoftmaxRows(const float* in, float* out, long long rows, long long cols) {
    // One warp per row.
    const int lane = threadIdx.x % 32;
    const long long row = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (row >= rows) return;
    const float* p = in + row * cols;
    float* q = out + row * cols;
    // Pass 1: each lane folds its strided elements into (m, s) in one sweep...
    float m = -FLT_MAX, s = 0.0f;
    for (long long j = lane; j < cols; j += 32) merge(m, s, p[j], 1.0f);
    // ...and a butterfly merges the 32 pairs, so every lane holds the row's max and sum.
    for (int o = 16; o > 0; o >>= 1) merge(m, s, __shfl_xor_sync(0xffffffffu, m, o), __shfl_xor_sync(0xffffffffu, s, o));
    // Pass 2: log-softmax = x - logsumexp, no second exponential needed.
    const float lse = m + logf(s);
    for (long long j = lane; j < cols; j += 32) q[j] = p[j] - lse;
}

// input, output are device pointers
extern "C" void solution(const float* input, float* output, size_t M, size_t N) {
    // 32 threads (one warp) per row.
    const long long threads = static_cast<long long>(M) * 32;
    logSoftmaxRows<<<static_cast<unsigned>((threads + kBlockSize - 1) / kBlockSize), kBlockSize>>>(input, output, M, N);
    cudaDeviceSynchronize();
}

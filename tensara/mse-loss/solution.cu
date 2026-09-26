// MSE Loss (Tensara)
// https://tensara.org/problems/mse-loss
//
// mean((pred - target)^2) over a tensor of arbitrary shape (element count =
// product of shape). Two-pass reduction with fp64 block partials. `shape` may be
// a host or device pointer, so it is read with cudaMemcpyDefault.
#include <cuda_runtime.h>

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 1024;

__device__ double g_partials[kMaxBlocks];

__device__ double blockSumD(double v) {
    __shared__ double warp_sums[32];
    __shared__ double total;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x % 32 == 0) warp_sums[threadIdx.x / 32] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        double t = 0.0;
        for (int w = 0; w < static_cast<int>(blockDim.x / 32); ++w) t += warp_sums[w];
        total = t;
    }
    __syncthreads();
    const double result = total;
    __syncthreads();
    return result;
}

__global__ void squaredDiffs(const float* p, const float* t, size_t n) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    float local = 0.0f;
    for (size_t i = tid; i < n; i += stride) {
        const float d = p[i] - t[i];
        local += d * d;
    }
    const double s = blockSumD(local);
    if (threadIdx.x == 0) g_partials[blockIdx.x] = s;
}

__global__ void finalize(float* out, int partials, size_t n) {
    double v = 0.0;
    for (int i = threadIdx.x; i < partials; i += blockDim.x) v += g_partials[i];
    v = blockSumD(v);
    if (threadIdx.x == 0) out[0] = static_cast<float>(v / n);
}

// predictions, targets, output are device pointers
extern "C" void solution(const float* predictions, const float* targets, float* output, const size_t* shape, size_t ndim) {
    size_t host_shape[16];
    cudaMemcpy(host_shape, shape, ndim * sizeof(size_t), cudaMemcpyDefault);
    size_t n = 1;
    for (size_t d = 0; d < ndim; ++d) n *= host_shape[d];
    size_t blocks = (n + kThreads - 1) / kThreads;
    blocks = blocks < 1 ? 1 : (blocks > kMaxBlocks ? kMaxBlocks : blocks);
    squaredDiffs<<<static_cast<unsigned>(blocks), kThreads>>>(predictions, targets, n);
    finalize<<<1, kThreads>>>(output, static_cast<int>(blocks), n);
    cudaDeviceSynchronize();
}

// Frobenius Norm (Tensara)
// https://tensara.org/problems/frobenius-norm
//
// y = x / ||x||_F over the whole tensor.
//   1. grid-stride partial sums of x^2 (float4 loads) -> fp64 block partials;
//   2. one block turns them into 1 / norm;
//   3. elementwise scale.
#include <cuda_runtime.h>

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 1024;

// Device-global scratch: one fp64 partial sum per block, and the final 1 / norm.
__device__ double g_partials[kMaxBlocks];
__device__ float g_inv_norm;

__device__ double blockSumD(double v) {
    // Block-wide sum in double; every thread receives the result.
    __shared__ double warp_sums[32];
    __shared__ double total;
    // Butterfly sum inside each warp; lane 0 publishes the warp total.
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x % 32 == 0) warp_sums[threadIdx.x / 32] = v;
    __syncthreads();
    // Thread 0 adds the warp totals and broadcasts the result through shared memory.
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

__global__ void sumSquares(const float* x, size_t n) {
    // Pass 1: grid-stride sum of squares with float4 loads, scalar tail for n % 4.
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    float local = 0.0f;
    for (size_t i = tid; i < n / 4; i += stride) {
        const float4 v = reinterpret_cast<const float4*>(x)[i];
        local += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (size_t i = (n / 4) * 4 + tid; i < n; i += stride) local += x[i] * x[i];
    // Block partial in double, written by thread 0.
    const double s = blockSumD(local);
    if (threadIdx.x == 0) g_partials[blockIdx.x] = s;
}

__global__ void finishNorm(int partials) {
    // Pass 2 (one block): add the partials in double and store 1 / sqrt(sum).
    double v = 0.0;
    for (int i = threadIdx.x; i < partials; i += blockDim.x) v += g_partials[i];
    v = blockSumD(v);
    if (threadIdx.x == 0) g_inv_norm = static_cast<float>(1.0 / sqrt(v));
}

__global__ void scale(const float* x, float* y, size_t n) {
    // Pass 3: scale every element by the reciprocal norm.
    const float s = g_inv_norm;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        y[i] = x[i] * s;
}

// X, Y are device pointers
extern "C" void solution(const float* X, float* Y, size_t size) {
    // Three launches on one stream: the kernel boundaries act as grid-wide barriers.
    size_t blocks = (size / 4 + kThreads - 1) / kThreads;
    blocks = blocks < 1 ? 1 : (blocks > kMaxBlocks ? kMaxBlocks : blocks);
    sumSquares<<<static_cast<unsigned>(blocks), kThreads>>>(X, size);
    finishNorm<<<1, kThreads>>>(static_cast<int>(blocks));
    scale<<<4096, kThreads>>>(X, Y, size);
}

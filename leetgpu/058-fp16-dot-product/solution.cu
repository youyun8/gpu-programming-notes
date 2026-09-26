// FP16 Dot Product (LeetGPU)
// https://leetgpu.com/challenges/fp16-dot-product
//
// Two-pass reduction: half2 loads (two elements per 32-bit load), products
// accumulated in fp32 per thread, fp64 block partials, one final block that
// converts the result to half.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 1024;

__device__ double g_partials[kMaxBlocks];

__device__ double blockReduceSum(double v) {
    __shared__ double warp_sums[32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    v = threadIdx.x < blockDim.x / 32 ? warp_sums[lane] : 0.0;
    if (warp == 0)
        for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__global__ void partialDots(const half* a, const half* b, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    float local = 0.0f;
    const half2* a2 = reinterpret_cast<const half2*>(a);
    const half2* b2 = reinterpret_cast<const half2*>(b);
    for (int i = tid; i < n / 2; i += stride) {
        const float2 x = __half22float2(a2[i]);
        const float2 y = __half22float2(b2[i]);
        local = fmaf(x.x, y.x, local);
        local = fmaf(x.y, y.y, local);
    }
    if (tid == 0 && (n & 1)) local = fmaf(__half2float(a[n - 1]), __half2float(b[n - 1]), local);
    const double s = blockReduceSum(local);
    if (threadIdx.x == 0) g_partials[blockIdx.x] = s;
}

__global__ void finalSum(half* result, int num_partials) {
    double v = 0.0;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) v += g_partials[i];
    v = blockReduceSum(v);
    if (threadIdx.x == 0) result[0] = __float2half(static_cast<float>(v));
}

// A, B, result are device pointers
extern "C" void solve(const half* A, const half* B, half* result, int N) {
    int blocks = (N / 2 + kBlockSize - 1) / kBlockSize;
    blocks = blocks < 1 ? 1 : (blocks > kMaxBlocks ? kMaxBlocks : blocks);
    partialDots<<<blocks, kBlockSize>>>(A, B, N);
    finalSum<<<1, kBlockSize>>>(result, blocks);
    cudaDeviceSynchronize();
}

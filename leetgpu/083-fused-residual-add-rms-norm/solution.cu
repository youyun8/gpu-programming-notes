// Fused Residual Add + RMSNorm (LeetGPU)
// https://leetgpu.com/challenges/fused-residual-add-rms-norm
//
// out = (x + r) / sqrt(mean((x + r)^2) + eps) * w, per row.
// One block per row; z = x + r is never written to global memory:
//   pass 1: accumulate sum(z^2) (float4 loads) and block-reduce it;
//   pass 2: recompute z (the row is hot in L1/L2) and write the output.
#include <cuda_runtime.h>

constexpr int kThreads = 256;

__device__ float blockSum(float v) {
    __shared__ float warp_sums[kThreads / 32];
    __shared__ float total;
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    if (threadIdx.x % 32 == 0) warp_sums[threadIdx.x / 32] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int w = 0; w < kThreads / 32; ++w) t += warp_sums[w];
        total = t;
    }
    __syncthreads();
    return total;
}

__global__ void residualRmsNorm(const float* x, const float* r, const float* w, float* out, int c, float eps) {
    const size_t base = static_cast<size_t>(blockIdx.x) * c;
    const float* xr = x + base;
    const float* rr = r + base;
    float* orow = out + base;
    const bool vec = (c % 4) == 0;  // rows are 16-byte aligned only when C % 4 == 0

    float sq = 0.0f;
    if (vec) {
        for (int i = threadIdx.x; i < c / 4; i += kThreads) {
            const float4 a = reinterpret_cast<const float4*>(xr)[i];
            const float4 b = reinterpret_cast<const float4*>(rr)[i];
            const float z0 = a.x + b.x, z1 = a.y + b.y, z2 = a.z + b.z, z3 = a.w + b.w;
            sq += z0 * z0 + z1 * z1 + z2 * z2 + z3 * z3;
        }
    } else {
        for (int i = threadIdx.x; i < c; i += kThreads) {
            const float z = xr[i] + rr[i];
            sq += z * z;
        }
    }
    const float inv_rms = rsqrtf(blockSum(sq) / c + eps);

    if (vec) {
        for (int i = threadIdx.x; i < c / 4; i += kThreads) {
            const float4 a = reinterpret_cast<const float4*>(xr)[i];
            const float4 b = reinterpret_cast<const float4*>(rr)[i];
            const float4 g = reinterpret_cast<const float4*>(w)[i];
            reinterpret_cast<float4*>(orow)[i] = make_float4((a.x + b.x) * inv_rms * g.x, (a.y + b.y) * inv_rms * g.y,
                                                             (a.z + b.z) * inv_rms * g.z, (a.w + b.w) * inv_rms * g.w);
        }
    } else {
        for (int i = threadIdx.x; i < c; i += kThreads) orow[i] = (xr[i] + rr[i]) * inv_rms * w[i];
    }
}

// x, residual, weight, out are device pointers
extern "C" void solve(const float* x, const float* residual, const float* weight, float* out, int N, int C, float eps) {
    residualRmsNorm<<<N, kThreads>>>(x, residual, weight, out, C, eps);
    cudaDeviceSynchronize();
}

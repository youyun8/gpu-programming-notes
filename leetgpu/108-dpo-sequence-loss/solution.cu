// DPO Sequence Loss (LeetGPU)
// https://leetgpu.com/challenges/dpo-sequence-loss
//
// loss = mean(softplus(-z)), z = beta ((l+ - l-) - (l+ref - l-ref)).
// softplus(-z) = max(-z, 0) + log1p(exp(-|z|)) is stable for any z.
// Elementwise transform fused into a single-block reduction (B is small).
#include <cuda_runtime.h>

constexpr int kThreads = 1024;

__global__ void dpoLoss(const float* lc, const float* lr, const float* lc_ref, const float* lr_ref, float* out, float beta,
                        int b) {
    __shared__ double warp_sums[kThreads / 32];
    double local = 0.0;
    for (int i = threadIdx.x; i < b; i += kThreads) {
        const float z = beta * ((lc[i] - lr[i]) - (lc_ref[i] - lr_ref[i]));
        local += fmaxf(-z, 0.0f) + log1pf(expf(-fabsf(z)));
    }
    for (int offset = 16; offset > 0; offset >>= 1) local += __shfl_xor_sync(0xffffffffu, local, offset);
    if (threadIdx.x % 32 == 0) warp_sums[threadIdx.x / 32] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        double t = 0.0;
        for (int w = 0; w < kThreads / 32; ++w) t += warp_sums[w];
        out[0] = static_cast<float>(t / b);
    }
}

// all pointers are device pointers
extern "C" void solve(const float* chosen_logps, const float* rejected_logps, const float* chosen_ref_logps,
                      const float* rejected_ref_logps, float* output, float beta, int B) {
    dpoLoss<<<1, kThreads>>>(chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, output, beta, B);
    cudaDeviceSynchronize();
}

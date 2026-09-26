// KL Divergence (elementwise) (Tensara)
// https://tensara.org/problems/kl-loss
//
// t * (log t - log p) with both clamped at 1e-10, and 0 where t <= 0.
// Pure elementwise map over two inputs: grid-stride loop, coalesced loads.
#include <cuda_runtime.h>

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

__global__ void lossKernel(const float* __restrict__ pred, const float* __restrict__ targ, float* __restrict__ out, size_t n) {
    // Grid-stride elementwise map over two inputs (coalesced loads).
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float p = pred[i];
        const float t = targ[i];
        // Reproduce the reference: clamp both inputs at 1e-10 before the logs,
        // and output 0 wherever the (unclamped) target is not positive.
        out[i] = t > 0.0f ? fmaxf(t, 1e-10f) * (logf(fmaxf(t, 1e-10f)) - logf(fmaxf(p, 1e-10f))) : 0.0f;
    }
}

// predictions, targets, output are device pointers
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
    size_t blocks = (n + kThreads - 1) / kThreads;
    blocks = blocks < 1 ? 1 : (blocks > kMaxBlocks ? kMaxBlocks : blocks);
    lossKernel<<<static_cast<unsigned>(blocks), kThreads>>>(predictions, targets, output, n);
}

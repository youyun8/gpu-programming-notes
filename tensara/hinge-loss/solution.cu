// Hinge Loss (Tensara)
// https://tensara.org/problems/hinge-loss
//
// max(0, 1 - p * t), elementwise.
// Pure elementwise map over two inputs: grid-stride loop, coalesced loads.
#include <cuda_runtime.h>

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

__global__ void lossKernel(const float* __restrict__ pred, const float* __restrict__ targ, float* __restrict__ out, size_t n) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float p = pred[i];
        const float t = targ[i];
        out[i] = fmaxf(1.0f - p * t, 0.0f);
    }
}

// predictions, targets, output are device pointers
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
    size_t blocks = (n + kThreads - 1) / kThreads;
    blocks = blocks < 1 ? 1 : (blocks > kMaxBlocks ? kMaxBlocks : blocks);
    lossKernel<<<static_cast<unsigned>(blocks), kThreads>>>(predictions, targets, output, n);
}

// Huber Loss (Tensara)
// https://tensara.org/problems/huber-loss
//
// Smooth L1 (beta = 1): 0.5 d^2 if abs(d) < 1, else abs(d) - 0.5, with d = p - t.
// Pure elementwise map over two inputs: grid-stride loop, coalesced loads.
#include <cuda_runtime.h>

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

__global__ void lossKernel(const float* __restrict__ pred, const float* __restrict__ targ, float* __restrict__ out, size_t n) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float p = pred[i];
        const float t = targ[i];
        const float d = p - t;
        const float ad = fabsf(d);
        out[i] = ad < 1.0f ? 0.5f * d * d : ad - 0.5f;
    }
}

// predictions, targets, output are device pointers
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
    size_t blocks = (n + kThreads - 1) / kThreads;
    blocks = blocks < 1 ? 1 : (blocks > kMaxBlocks ? kMaxBlocks : blocks);
    lossKernel<<<static_cast<unsigned>(blocks), kThreads>>>(predictions, targets, output, n);
}

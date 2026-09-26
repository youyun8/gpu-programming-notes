// Softmax Attention Backward (LeetGPU)
// https://leetgpu.com/challenges/softmax-attention-backward
#include <cuda_runtime.h>

extern "C" void solve(const float* Q, const float* K, const float* V, const float* dO, float* dQ, float* dK, float* dV, int M, int N, int d) {
}

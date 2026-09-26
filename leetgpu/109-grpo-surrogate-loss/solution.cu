// GRPO Surrogate Loss (LeetGPU)
// https://leetgpu.com/challenges/grpo-surrogate-loss
#include <cuda_runtime.h>

extern "C" void solve(const float* rewards, const float* log_pi, const float* log_pi_old, const float* log_ref, float* output, float clip_eps, float beta, int B, int G, int S) {
}

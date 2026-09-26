// Speculative Decoding Verification (LeetGPU)
// https://leetgpu.com/challenges/speculative-decoding-verification
#include <cuda_runtime.h>

extern "C" void solve(const int* draft_tokens, const float* draft_probs, const float* target_probs, const float* uniform_samples, int* output_tokens, int B, int T, int V) {
}

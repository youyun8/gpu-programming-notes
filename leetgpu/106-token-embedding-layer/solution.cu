// Token Embedding Layer (LeetGPU)
// https://leetgpu.com/challenges/token-embedding-layer
#include <cuda_runtime.h>

extern "C" void solve(const int* token_ids, const int* position_ids, const float* token_embeddings, const float* position_embeddings, const float* gamma, const float* beta, float* output, int B, int T, int V, int P, int D, float eps) {
}

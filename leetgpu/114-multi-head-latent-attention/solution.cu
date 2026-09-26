// Multi-Head Latent Attention Decode (LeetGPU)
// https://leetgpu.com/challenges/multi-head-latent-attention-decode
#include <cuda_runtime.h>

extern "C" void solve(const float* q, const float* kv_cache, const float* W_UK, const float* W_UV, float* output, int num_heads, int seq_len, int kv_lora_rank, int head_dim, int rope_dim) {
}

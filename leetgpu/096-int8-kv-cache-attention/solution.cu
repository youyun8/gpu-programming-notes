// INT8 KV-Cache Attention (LeetGPU)
// https://leetgpu.com/challenges/int8-kv-cache-attention
#include <cuda_runtime.h>

extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8, const float* k_scale, const float* v_scale, float* output, int num_heads, int seq_len, int head_dim) {
}

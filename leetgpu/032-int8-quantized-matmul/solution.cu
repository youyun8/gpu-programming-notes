// INT8 Quantized MatMul (LeetGPU)
// https://leetgpu.com/challenges/int8-quantized-matmul
#include <cuda_runtime.h>

extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, float scale_A, float scale_B, float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {
}

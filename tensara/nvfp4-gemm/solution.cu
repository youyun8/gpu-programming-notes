// NVFP4 GEMM (Tensara)
// https://tensara.org/problems/nvfp4-gemm
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

extern "C" void solution(const uint8_t* q_a, const uint8_t* scale_a, const float sf_g_a, const uint8_t* q_b, const uint8_t* scale_b, const float sf_g_b, __half* c, size_t m, size_t n, size_t k) {
}

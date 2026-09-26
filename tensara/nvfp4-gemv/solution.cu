// NVFP4 GEMV (Tensara)
// https://tensara.org/problems/nvfp4-gemv
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

extern "C" void solution(const uint8_t* q_a, const uint8_t* scale_a, const float sf_g_a, const uint8_t* q_x, const uint8_t* scale_x, const float sf_g_x, __half* y, size_t m, size_t k) {
}

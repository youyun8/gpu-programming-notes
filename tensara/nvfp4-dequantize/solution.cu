// NVFP4 Dequantization (Tensara)
// https://tensara.org/problems/nvfp4-dequantize
#include <cuda_runtime.h>
#include <cstdint>

extern "C" void solution(const uint8_t* q, const uint8_t* scale, const float sf_g, float* out, size_t m, size_t k) {
}

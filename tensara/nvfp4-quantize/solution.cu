// NVFP4 Quantization (Tensara)
// https://tensara.org/problems/nvfp4-quantize
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

extern "C" void solution(const __half* a, const float sf_g, uint8_t* q, uint8_t* scale, size_t m, size_t k) {
}

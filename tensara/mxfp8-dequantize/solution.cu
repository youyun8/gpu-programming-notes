// MXFP8 Dequantization (Tensara)
// https://tensara.org/problems/mxfp8-dequantize
#include <cuda_runtime.h>
#include <cstdint>

extern "C" void solution(const uint8_t* q, const uint8_t* scale, float* out, size_t m, size_t k) {
}

// Triplet Margin Loss (Tensara)
// https://tensara.org/problems/triplet-margin
#include <cuda_runtime.h>

extern "C" void solution(const float* anchor, const float* positive, const float* negative, float* loss, size_t B, size_t E, float margin) {
}

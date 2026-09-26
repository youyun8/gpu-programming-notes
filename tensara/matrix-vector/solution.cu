// Matrix-Vector Multiplication (Tensara)
// https://tensara.org/problems/matrix-vector
//
// c = A b, A: M x K. GEMV is bandwidth-bound (each A element used once), so
// the goal is streaming A at full bandwidth: one warp per row, float4 loads
// when rows are 16-byte aligned (K % 4 == 0), b stays in L1/L2, shuffle
// reduction per row.
#include <cuda_runtime.h>

constexpr int kWarpsPerBlock = 8;

__global__ void gemv(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t m, size_t k) {
    const int lane = threadIdx.x % 32;
    const size_t row = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / 32;
    if (row >= m) return;
    const float* ar = a + row * k;
    float sum = 0.0f;
    if (k % 4 == 0) {
        const float4* a4 = reinterpret_cast<const float4*>(ar);
        const float4* b4 = reinterpret_cast<const float4*>(b);
        for (size_t j = lane; j < k / 4; j += 32) {
            const float4 x = a4[j], y = b4[j];
            sum += x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
        }
    } else {
        for (size_t j = lane; j < k; j += 32) sum = fmaf(ar[j], b[j], sum);
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
    if (lane == 0) c[row] = sum;
}

// input_a, input_b, output_c are device pointers
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t k) {
    gemv<<<static_cast<unsigned>((m + kWarpsPerBlock - 1) / kWarpsPerBlock), kWarpsPerBlock * 32>>>(input_a, input_b, output_c, m, k);
}

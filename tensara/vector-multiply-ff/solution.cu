// Vector Multiply over F_p (Tensara)
// https://tensara.org/problems/vector-multiply-ff
//
// c_i = a_i * b_i mod p, p = 2^31 - 1 (a Mersenne prime).
// Because 2^31 = 1 (mod p), a 62-bit product x = hi * 2^31 + lo reduces to
// hi + lo with shifts and masks - no division: fold twice, subtract p once.
#include <cstdint>
#include <cuda_runtime.h>

constexpr uint64_t kP = (1ull << 31) - 1;

__device__ __forceinline__ uint32_t mulModMersenne31(uint32_t a, uint32_t b) {
    // Mersenne reduction: 2^31 = 1 (mod p), so x = hi * 2^31 + lo is congruent to hi + lo.
    // Two folds bring x to at most p + 1, one conditional subtraction finishes.
    uint64_t x = static_cast<uint64_t>(a) * b;  // < 2^62
    x = (x & kP) + (x >> 31);                   // < 2^32
    x = (x & kP) + (x >> 31);                   // <= p + 1
    return static_cast<uint32_t>(x >= kP ? x - kP : x);
}

__global__ void mulMod(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b, uint32_t* __restrict__ c, size_t n) {
    // Grid-stride elementwise map.
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        c[i] = mulModMersenne31(a[i], b[i]);
}

// d_input1, d_input2, d_output are device pointers
extern "C" void solution(const uint32_t* d_input1, const uint32_t* d_input2, uint32_t* d_output, size_t n) {
    // One thread per element, capped grid.
    size_t blocks = (n + 255) / 256;
    blocks = blocks > 4096 ? 4096 : (blocks < 1 ? 1 : blocks);
    mulMod<<<static_cast<unsigned>(blocks), 256>>>(d_input1, d_input2, d_output, n);
}

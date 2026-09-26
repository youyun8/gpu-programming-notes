// Polynomial Multiplication over F_p (Tensara)
// https://tensara.org/problems/poly-multiply-ff
//
// c_k = sum_{i + j = k} a_i b_j mod p, p = 2^31 - 1, output length 2n - 1.
// p - 1 has no large power-of-two factor, so a direct NTT over F_p is not
// available; for the tested sizes (n <= 1024) the O(n^2) convolution is fast:
//   - one thread per output coefficient, a and b staged in shared memory in
//     1024-element tiles (b read reversed, a broadcast-free sequential sweep);
//   - every product is reduced with the Mersenne fold (no division) and the
//     reduced terms (< 2^31) are summed in 64 bits (1024 terms < 2^41), with a
//     final fold at the end.
#include <cstdint>
#include <cuda_runtime.h>

constexpr uint64_t kP = (1ull << 31) - 1;
constexpr int kThreads = 256;
constexpr int kTile = 1024;

__device__ __forceinline__ uint64_t foldMersenne31(uint64_t x) {
    x = (x & kP) + (x >> 31);
    x = (x & kP) + (x >> 31);
    return x >= kP ? x - kP : x;
}

__global__ void polyMul(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b, uint32_t* __restrict__ c, int n) {
    __shared__ uint32_t s_a[kTile];
    __shared__ uint32_t s_b[kTile];
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t acc = 0;
    for (int i0 = 0; i0 < n; i0 += kTile) {
        for (int j0 = 0; j0 < n; j0 += kTile) {
            __syncthreads();
            for (int t = threadIdx.x; t < kTile; t += kThreads) {
                s_a[t] = i0 + t < n ? a[i0 + t] : 0u;
                s_b[t] = j0 + t < n ? b[j0 + t] : 0u;
            }
            __syncthreads();
            if (k < 2 * n - 1) {
                // i in [i0, i0 + kTile), j = k - i in [j0, j0 + kTile)
                const int lo = max(i0, k - (j0 + kTile - 1));
                const int hi = min(i0 + kTile - 1, k - j0);
                for (int i = lo; i <= hi; ++i) {
                    acc += foldMersenne31(static_cast<uint64_t>(s_a[i - i0]) * s_b[k - i - j0]);
                    if (acc >= (1ull << 62)) acc = foldMersenne31(acc);
                }
            }
        }
    }
    if (k < 2 * n - 1) c[k] = static_cast<uint32_t>(foldMersenne31(acc));
}

// d_input1, d_input2, d_output are device pointers
extern "C" void solution(const uint32_t* d_input1, const uint32_t* d_input2, uint32_t* d_output, size_t n) {
    const int out_len = 2 * static_cast<int>(n) - 1;
    polyMul<<<(out_len + kThreads - 1) / kThreads, kThreads>>>(d_input1, d_input2, d_output, static_cast<int>(n));
}

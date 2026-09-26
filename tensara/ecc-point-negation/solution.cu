// ECC Point Negation (Tensara)
// https://tensara.org/problems/ecc-point-negation
//
// -(x, y) = (x, (p - y mod p) mod p) over F_p, p = 2^61 - 1, output interleaved
// as out[2i] = x_i, out[2i + 1] = -y_i. Pure bandwidth: each thread reads one
// x and one y and writes both results as a single 16-byte store.
#include <cstdint>
#include <cuda_runtime.h>

__global__ void negatePoints(const uint64_t* __restrict__ xs, const uint64_t* __restrict__ ys, uint64_t p, ulonglong2* __restrict__ out, size_t n) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        // -(x, y) = (x, -y mod p); y = 0 must map to 0, not p.
        const uint64_t y = ys[i] % p;
        // Write (x, -y) as one 16-byte store into the interleaved output.
        out[i] = make_ulonglong2(xs[i], y == 0 ? 0 : p - y);
    }
}

// xs, ys, out_xy are device pointers
extern "C" void solution(const uint64_t* xs, const uint64_t* ys, const uint64_t p, uint64_t* out_xy, size_t n) {
    // One thread per point, grid capped at 4096 blocks (grid-stride loop).
    size_t blocks = (n + 255) / 256;
    blocks = blocks > 4096 ? 4096 : (blocks < 1 ? 1 : blocks);
    negatePoints<<<static_cast<unsigned>(blocks), 256>>>(xs, ys, p, reinterpret_cast<ulonglong2*>(out_xy), n);
}

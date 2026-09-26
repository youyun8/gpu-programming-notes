// Diagonal Matrix Multiplication (Tensara)
// https://tensara.org/problems/diagonal-matmul
//
// C = diag(a) B scales row i of B by a[i]: an elementwise, bandwidth-bound
// kernel (never form the N x N diagonal matrix). 2-D grid: threadIdx.x along
// the columns for coalescing; a[i] is a broadcast.
#include <cuda_runtime.h>

__global__ void scaleRows(const float* __restrict__ diag, const float* __restrict__ b, float* __restrict__ c, size_t n, size_t m) {
    // 2-D grid: x covers 256 columns (coalesced), y walks the rows with a grid stride.
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (size_t row = blockIdx.y; row < n; row += gridDim.y) {
        // diag(a) * B only scales row `row` by a[row]; a[row] is the same for the whole block (a broadcast).
        if (col < m) c[row * m + col] = diag[row] * b[row * m + col];
    }
}

// diagonal_a, input_b, output_c are device pointers
extern "C" void solution(const float* diagonal_a, const float* input_b, float* output_c, size_t n, size_t m) {
    const dim3 block(256);
    // Never materialize the N x N diagonal matrix: this is an elementwise kernel.
    const dim3 grid(static_cast<unsigned>((m + 255) / 256), static_cast<unsigned>(n < 65535 ? n : 65535));
    scaleRows<<<grid, block>>>(diagonal_a, input_b, output_c, n, m);
}

// Jacobi Stencil 2D (LeetGPU)
// https://leetgpu.com/challenges/2d-jacobi-stencil
//
// One sweep of the 5-point stencil; boundary cells are copied. A 32 x 8 block
// with threadIdx.x along columns: the left/right/center reads of a warp share
// cache lines and the rows above/below are reused by neighbouring blocks from
// L2, so the kernel runs at close to copy bandwidth without explicit tiling.
#include <cuda_runtime.h>

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;

__global__ void jacobi(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    // One thread per cell: 32 x 8 blocks, x along the contiguous columns.
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;
    const size_t idx = static_cast<size_t>(r) * cols + c;
    // Boundary cells are copied; interior cells get the average of their 4 neighbours,
    // summed in the reference's order (up + down + left + right).
    if (r == 0 || c == 0 || r == rows - 1 || c == cols - 1) {
        out[idx] = in[idx];
    } else {
        out[idx] = 0.25f * (((in[idx - cols] + in[idx + cols]) + in[idx - 1]) + in[idx + 1]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    // One sweep: read `input`, write `output` (no in-place update, so no races).
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid((cols + kBlockX - 1) / kBlockX, (rows + kBlockY - 1) / kBlockY);
    jacobi<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

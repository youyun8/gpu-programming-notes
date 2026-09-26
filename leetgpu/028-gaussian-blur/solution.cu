// Gaussian Blur (LeetGPU)
// https://leetgpu.com/challenges/gaussian-blur
//
// "Same" 2D convolution with zero padding, odd kernel up to 21 x 21.
// A 32 x 8 block computes a 32 x 32 output tile; the input window including
// the halo (kernel_rows/2, kernel_cols/2 on each side) and the kernel are
// staged in dynamic shared memory with out-of-image pixels set to zero.
#include <cuda_runtime.h>

constexpr int kTileX = 32;
constexpr int kTileY = 32;
constexpr int kBlockY = 8;
constexpr int kRowsPerThread = kTileY / kBlockY;

__global__ void gaussianBlur(const float* input, const float* kernel, float* output, int rows, int cols, int k_rows,
                             int k_cols) {
    extern __shared__ float smem[];
    const int win_cols = kTileX + k_cols - 1;
    const int win_rows = kTileY + k_rows - 1;
    float* s_kernel = smem;
    float* s_input = smem + k_rows * k_cols;

    const int row0 = blockIdx.y * kTileY - k_rows / 2;
    const int col0 = blockIdx.x * kTileX - k_cols / 2;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * blockDim.y;

    for (int i = tid; i < k_rows * k_cols; i += num_threads) s_kernel[i] = kernel[i];
    for (int i = tid; i < win_rows * win_cols; i += num_threads) {
        const int r = row0 + i / win_cols;
        const int c = col0 + i % win_cols;
        s_input[i] = (r >= 0 && r < rows && c >= 0 && c < cols) ? input[static_cast<size_t>(r) * cols + c] : 0.0f;
    }
    __syncthreads();

    float acc[kRowsPerThread] = {};
    for (int kr = 0; kr < k_rows; ++kr) {
        for (int kc = 0; kc < k_cols; ++kc) {
            const float w = s_kernel[kr * k_cols + kc];
#pragma unroll
            for (int r = 0; r < kRowsPerThread; ++r) {
                acc[r] = fmaf(s_input[(threadIdx.y + r * kBlockY + kr) * win_cols + threadIdx.x + kc], w, acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        const int orow = blockIdx.y * kTileY + threadIdx.y + r * kBlockY;
        const int ocol = blockIdx.x * kTileX + threadIdx.x;
        if (orow < rows && ocol < cols) output[static_cast<size_t>(orow) * cols + ocol] = acc[r];
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows, int input_cols,
                      int kernel_rows, int kernel_cols) {
    const dim3 block(kTileX, kBlockY);
    const dim3 grid((input_cols + kTileX - 1) / kTileX, (input_rows + kTileY - 1) / kTileY);
    const size_t smem = (static_cast<size_t>(kernel_rows) * kernel_cols +
                         static_cast<size_t>(kTileY + kernel_rows - 1) * (kTileX + kernel_cols - 1)) * sizeof(float);
    gaussianBlur<<<grid, block, smem>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}

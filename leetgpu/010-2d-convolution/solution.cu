// 2D Convolution (LeetGPU)
// https://leetgpu.com/challenges/2d-convolution
//
// "Valid" 2D cross-correlation, kernel up to 31 x 31.
// A 32 x 8 block computes a 32 x 32 output tile (4 rows per thread). The input
// window ((32 + kr - 1) x (32 + kc - 1)) and the kernel live in dynamic shared
// memory, so each input element is fetched from DRAM ~once per tile.
#include <cuda_runtime.h>

constexpr int kTileX = 32;
constexpr int kTileY = 32;
constexpr int kBlockY = 8;
constexpr int kRowsPerThread = kTileY / kBlockY;

__global__ void conv2d(const float* input, const float* kernel, float* output, int in_rows, int in_cols, int k_rows,
                       int k_cols) {
    // Dynamic shared memory: the whole kernel, then the input window of this output tile.
    extern __shared__ float smem[];
    const int win_cols = kTileX + k_cols - 1;
    const int win_rows = kTileY + k_rows - 1;
    float* s_kernel = smem;
    float* s_input = smem + k_rows * k_cols;

    // "Valid" convolution: a 32 x 32 output tile needs a (32 + k_rows - 1) x (32 + k_cols - 1) window.
    const int out_rows = in_rows - k_rows + 1;
    const int out_cols = in_cols - k_cols + 1;
    const int row0 = blockIdx.y * kTileY;
    const int col0 = blockIdx.x * kTileX;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * blockDim.y;

    // Stage the kernel and the window (zero outside the input).
    for (int i = tid; i < k_rows * k_cols; i += num_threads) s_kernel[i] = kernel[i];
    for (int i = tid; i < win_rows * win_cols; i += num_threads) {
        const int r = row0 + i / win_cols;
        const int c = col0 + i % win_cols;
        s_input[i] = (r < in_rows && c < in_cols) ? input[static_cast<size_t>(r) * in_cols + c] : 0.0f;
    }
    // Staged data visible to all threads.
    __syncthreads();

    // Each thread computes 4 output rows (threadIdx.y + 8r) of one column; each weight is a
    // broadcast and lanes read consecutive window columns.
    float acc[kRowsPerThread] = {};
    for (int kr = 0; kr < k_rows; ++kr) {
        for (int kc = 0; kc < k_cols; ++kc) {
            const float w = s_kernel[kr * k_cols + kc];
            // Store in bounds.
#pragma unroll
            for (int r = 0; r < kRowsPerThread; ++r) {
                const int local_row = threadIdx.y + r * kBlockY + kr;
                acc[r] = fmaf(s_input[local_row * win_cols + threadIdx.x + kc], w, acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        const int orow = row0 + threadIdx.y + r * kBlockY;
        const int ocol = col0 + threadIdx.x;
        if (orow < out_rows && ocol < out_cols) output[static_cast<size_t>(orow) * out_cols + ocol] = acc[r];
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows, int input_cols,
                      int kernel_rows, int kernel_cols) {
    const int out_rows = input_rows - kernel_rows + 1;
    const int out_cols = input_cols - kernel_cols + 1;
    // 32 x 8 threads per 32 x 32 output tile; shared memory sized for the kernel and the window.
    const dim3 block(kTileX, kBlockY);
    const dim3 grid((out_cols + kTileX - 1) / kTileX, (out_rows + kTileY - 1) / kTileY);
    const size_t smem = (static_cast<size_t>(kernel_rows) * kernel_cols +
                         static_cast<size_t>(kTileY + kernel_rows - 1) * (kTileX + kernel_cols - 1)) * sizeof(float);
    conv2d<<<grid, block, smem>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}

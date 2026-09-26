// 3D Convolution (LeetGPU)
// https://leetgpu.com/challenges/3d-convolution
//
// "Valid" 3D cross-correlation, kernel <= 5 x 5 x 5, volume <= 256^3.
// One thread per output voxel; threadIdx.x runs along columns so neighbouring
// threads read neighbouring input words (coalesced, mostly L1 hits across the
// kernel window). The kernel (<= 125 taps) is staged in shared memory where
// every access is a broadcast.
#include <cuda_runtime.h>

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kMaxTaps = 125;

__global__ void conv3d(const float* input, const float* kernel, float* output, int in_d, int in_r, int in_c, int k_d,
                       int k_r, int k_c) {
    // Stage the (at most 5 x 5 x 5) kernel in shared memory; reads are broadcasts.
    __shared__ float s_kernel[kMaxTaps];
    const int taps = k_d * k_r * k_c;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < taps; i += blockDim.x * blockDim.y) s_kernel[i] = kernel[i];
    __syncthreads();

    // One thread per output voxel: column from x (contiguous, coalesced), row from y, depth from z.
    const int out_r = in_r - k_r + 1;
    const int out_c = in_c - k_c + 1;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (c >= out_c || r >= out_r) return;

    // "Valid" convolution: every tap is in bounds, so no checks in the loops.
    float acc = 0.0f;
    for (int dz = 0; dz < k_d; ++dz) {
        for (int dr = 0; dr < k_r; ++dr) {
            const float* in_row = input + (static_cast<size_t>(z + dz) * in_r + (r + dr)) * in_c + c;
            const float* k_row = s_kernel + (dz * k_r + dr) * k_c;
            for (int dc = 0; dc < k_c; ++dc) acc = fmaf(in_row[dc], k_row[dc], acc);
        }
    }
    output[(static_cast<size_t>(z) * out_r + r) * out_c + c] = acc;
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_depth, int input_rows,
                      int input_cols, int kernel_depth, int kernel_rows, int kernel_cols) {
    const int out_d = input_depth - kernel_depth + 1;
    const int out_r = input_rows - kernel_rows + 1;
    const int out_c = input_cols - kernel_cols + 1;
    // 32 x 8 blocks per output plane, one plane per blockIdx.z.
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid((out_c + kBlockX - 1) / kBlockX, (out_r + kBlockY - 1) / kBlockY, out_d);
    conv3d<<<grid, block>>>(input, kernel, output, input_depth, input_rows, input_cols, kernel_depth, kernel_rows,
                            kernel_cols);
    cudaDeviceSynchronize();
}

// Matrix Addition (LeetGPU)
// https://leetgpu.com/challenges/matrix-addition
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;

// The N x N matrices are contiguous, so this is a flat vector add over N*N
// elements: float4 for the bulk, scalar for the tail.
__global__ void matrixAdd(const float* a, const float* b, float* c, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_vec4 = total / 4;
    if (idx < num_vec4) {
        const float4 x = reinterpret_cast<const float4*>(a)[idx];
        const float4 y = reinterpret_cast<const float4*>(b)[idx];
        reinterpret_cast<float4*>(c)[idx] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
    const int tail = num_vec4 * 4 + idx;
    if (idx < total % 4) c[tail] = a[tail] + b[tail];
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    const int total = N * N;
    const int num_blocks = (total / 4 + kBlockSize) / kBlockSize;
    matrixAdd<<<num_blocks, kBlockSize>>>(A, B, C, total);
    cudaDeviceSynchronize();
}

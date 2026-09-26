// Vector Addition (Tensara)
// https://tensara.org/problems/vector-addition
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 1024;

__global__ void vectorAddVec4(const float* input1, const float* input2, float* output, size_t n) {
    const size_t num_vec4 = n / 4;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const float4* a4 = reinterpret_cast<const float4*>(input1);
    const float4* b4 = reinterpret_cast<const float4*>(input2);
    float4* c4 = reinterpret_cast<float4*>(output);

    for (size_t i = start; i < num_vec4; i += stride) {
        const float4 a = a4[i];
        const float4 b = b4[i];
        c4[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }

    // Scalar tail for the last n % 4 elements.
    for (size_t i = num_vec4 * 4 + start; i < n; i += stride) {
        output[i] = input1[i] + input2[i];
    }
}

// input1, input2, output are device pointers
extern "C" void solution(const float* input1, const float* input2, float* output, size_t n) {
    const size_t num_vec4 = (n + 3) / 4;
    size_t num_blocks = (num_vec4 + kBlockSize - 1) / kBlockSize;
    if (num_blocks > kMaxBlocks) num_blocks = kMaxBlocks;
    if (num_blocks == 0) num_blocks = 1;
    vectorAddVec4<<<static_cast<unsigned int>(num_blocks), kBlockSize>>>(input1, input2, output, n);
}

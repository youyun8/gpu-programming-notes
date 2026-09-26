// Vector Addition (Tensara)
// https://tensara.org/problems/vector-addition
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 4096;

// float4 vectorized, grid-stride; scalar loop handles the n % 4 tail.
__global__ void vectorAddVec4(const float* a, const float* b, float* c, size_t n) {
    // Grid-stride setup; the buffers are viewed as float4 (cudaMalloc returns 256-byte-aligned pointers).
    const size_t num_vec4 = n / 4;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* c4 = reinterpret_cast<float4*>(c);
    // Main part: one 16-byte load from each input and one 16-byte store per iteration.
    for (size_t i = start; i < num_vec4; i += stride) {
        const float4 x = a4[i];
        const float4 y = b4[i];
        c4[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
    // Scalar tail for the last n % 4 elements.
    for (size_t i = num_vec4 * 4 + start; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// d_input1, d_input2, d_output are device pointers
extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    // One thread per float4, capped at kMaxBlocks blocks; all indices are 64-bit (n up to 2^30).
    size_t num_blocks = (n / 4 + kBlockSize - 1) / kBlockSize;
    num_blocks = num_blocks < 1 ? 1 : (num_blocks > kMaxBlocks ? kMaxBlocks : num_blocks);
    vectorAddVec4<<<static_cast<unsigned>(num_blocks), kBlockSize>>>(d_input1, d_input2, d_output, n);
}

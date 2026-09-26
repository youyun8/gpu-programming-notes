// Swish (Tensara)
// https://tensara.org/problems/swish
//
// swish(x) = x * sigmoid(x).
// Memory-bound elementwise kernel: grid-stride loop over float4 (16-byte
// loads/stores, fewer instructions per byte) plus a scalar tail for the last
// count % 4 elements. cudaMalloc'd buffers are 256-byte aligned, so the float4
// reinterpretation is safe.
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 4096;

__device__ __forceinline__ float op(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void elementwise(const float* __restrict__ in, float* __restrict__ out, size_t count) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t num_vec4 = count / 4;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);
    for (size_t i = tid; i < num_vec4; i += stride) {
        const float4 v = in4[i];
        out4[i] = make_float4(op(v.x), op(v.y), op(v.z), op(v.w));
    }
    for (size_t i = num_vec4 * 4 + tid; i < count; i += stride) out[i] = op(in[i]);
}

// input, output are device pointers
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    const size_t count = n * m;
    size_t blocks = (count / 4 + kBlockSize - 1) / kBlockSize;
    blocks = blocks < 1 ? 1 : (blocks > kMaxBlocks ? kMaxBlocks : blocks);
    elementwise<<<static_cast<unsigned>(blocks), kBlockSize>>>(input, output, count);
}

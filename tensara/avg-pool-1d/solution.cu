// Average Pooling 1D (Tensara)
// https://tensara.org/problems/avg-pool-1d
//
// avg_pool1d with zero padding and count_include_pad=True (PyTorch's default):
// every window is divided by kernel_size^1, padded positions contribute 0.
// One thread per output element (grid-stride), consecutive threads produce
// consecutive outputs of the innermost dimension, so window reads overlap in
// cache across the warp.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kThreads = 256;

__global__ void pool1d(const float* __restrict__ in, float* __restrict__ out, int h_in, int h_out, int k, int s, int p) {
    const size_t total = static_cast<size_t>(h_out);
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int h_pos = static_cast<int>(idx % h_out);
        float acc = 0.0f;
        for (int ih = 0; ih < k; ++ih) {
            const int xh = h_pos * s - p + ih;
            if (xh < 0 || xh >= h_in) continue;
            const float v = in[static_cast<size_t>(xh)];
            acc = acc + v;
        }
        out[idx] = acc / static_cast<float>(k);
    }
}

// input, output are device pointers
extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {
    const int h_out = (static_cast<int>(H) + 2 * padding - (kernel_size)) / stride + 1;
    const size_t total = static_cast<size_t>(h_out);
    size_t blocks = (total + kThreads - 1) / kThreads;
    blocks = blocks < 1 ? 1 : (blocks > 65535 ? 65535 : blocks);
    pool1d<<<static_cast<unsigned>(blocks), kThreads>>>(input, output, static_cast<int>(H), h_out, kernel_size, stride, padding);
}

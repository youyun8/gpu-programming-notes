// Average Pooling 3D (Tensara)
// https://tensara.org/problems/avg-pool-3d
//
// avg_pool3d with zero padding and count_include_pad=True (PyTorch's default):
// every window is divided by kernel_size^3, padded positions contribute 0.
// One thread per output element (grid-stride), consecutive threads produce
// consecutive outputs of the innermost dimension, so window reads overlap in
// cache across the warp.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kThreads = 256;

__global__ void pool3d(const float* __restrict__ in, float* __restrict__ out, int h_in, int h_out, int w_in, int w_out, int d_in, int d_out, int k, int s, int p) {
    const size_t total = static_cast<size_t>(h_out) * static_cast<size_t>(w_out) * static_cast<size_t>(d_out);
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int d_pos = static_cast<int>(idx % d_out);
        const int w_pos = static_cast<int>((idx / d_out) % w_out);
        const int h_pos = static_cast<int>(((idx / d_out) / w_out) % h_out);
        float acc = 0.0f;
        for (int ih = 0; ih < k; ++ih) {
            const int xh = h_pos * s - p + ih;
            if (xh < 0 || xh >= h_in) continue;
            for (int iw = 0; iw < k; ++iw) {
                const int xw = w_pos * s - p + iw;
                if (xw < 0 || xw >= w_in) continue;
                for (int id = 0; id < k; ++id) {
                    const int xd = d_pos * s - p + id;
                    if (xd < 0 || xd >= d_in) continue;
                    const float v = in[((xh) * static_cast<size_t>(w_in) + xw) * static_cast<size_t>(d_in) + xd];
                    acc = acc + v;
                }
            }
        }
        out[idx] = acc / static_cast<float>(k * k * k);
    }
}

// input, output are device pointers
extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H, size_t W, size_t D) {
    const int h_out = (static_cast<int>(H) + 2 * padding - (kernel_size)) / stride + 1;
    const int w_out = (static_cast<int>(W) + 2 * padding - (kernel_size)) / stride + 1;
    const int d_out = (static_cast<int>(D) + 2 * padding - (kernel_size)) / stride + 1;
    const size_t total = static_cast<size_t>(h_out) * static_cast<size_t>(w_out) * static_cast<size_t>(d_out);
    size_t blocks = (total + kThreads - 1) / kThreads;
    blocks = blocks < 1 ? 1 : (blocks > 65535 ? 65535 : blocks);
    pool3d<<<static_cast<unsigned>(blocks), kThreads>>>(input, output, static_cast<int>(H), h_out, static_cast<int>(W), w_out, static_cast<int>(D), d_out, kernel_size, stride, padding);
}

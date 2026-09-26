// Max Pooling 3D (Tensara)
// https://tensara.org/problems/max-pool-3d
//
// max_pool3d with padding (padded positions act as -inf) and dilation:
// window element i sits at out_pos * stride - padding + i * dilation.
// One thread per output element (grid-stride), consecutive threads produce
// consecutive outputs of the innermost dimension, so window reads overlap in
// cache across the warp.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kThreads = 256;

__global__ void pool3d(const float* __restrict__ in, float* __restrict__ out, int h_in, int h_out, int w_in, int w_out, int d_in, int d_out, int k, int s, int p, int dil) {
    const size_t total = static_cast<size_t>(h_out) * static_cast<size_t>(w_out) * static_cast<size_t>(d_out);
    // One thread per output element (grid-stride); consecutive threads produce neighbouring
    // outputs, so their windows overlap and the loads hit in L1/L2.
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        // Decode the flat index into output (h, w, d); d is the fastest (contiguous) axis.
        const int d_pos = static_cast<int>(idx % d_out);
        const int w_pos = static_cast<int>((idx / d_out) % w_out);
        const int h_pos = static_cast<int>(((idx / d_out) / w_out) % h_out);
        // Running maximum; padded taps are skipped, i.e. treated as -inf.
        float acc = -FLT_MAX;
        // Input position of each tap: out_pos * stride - padding + tap * dilation; skip taps outside the input.
        for (int ih = 0; ih < k; ++ih) {
            const int xh = h_pos * s - p + ih * dil;
            if (xh < 0 || xh >= h_in) continue;
            for (int iw = 0; iw < k; ++iw) {
                const int xw = w_pos * s - p + iw * dil;
                if (xw < 0 || xw >= w_in) continue;
                for (int id = 0; id < k; ++id) {
                    const int xd = d_pos * s - p + id * dil;
                    if (xd < 0 || xd >= d_in) continue;
                    const float v = in[((xh) * static_cast<size_t>(w_in) + xw) * static_cast<size_t>(d_in) + xd];
                    acc = fmaxf(acc, v);
                }
            }
        }
        // Store the window maximum (exact, so bit-identical to PyTorch).
        out[idx] = acc;
    }
}

// input, output are device pointers
extern "C" void solution(const float* input, int kernel_size, int stride, int padding, int dilation, float* output, size_t H, size_t W, size_t D) {
    // Output extent per axis: floor((X + 2P - dilation * (k - 1) - 1) / S) + 1.
    const int h_out = (static_cast<int>(H) + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1;
    const int w_out = (static_cast<int>(W) + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1;
    const int d_out = (static_cast<int>(D) + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1;
    const size_t total = static_cast<size_t>(h_out) * static_cast<size_t>(w_out) * static_cast<size_t>(d_out);
    // One thread per output, grid capped at 65535 blocks (the grid-stride loop covers the rest).
    size_t blocks = (total + kThreads - 1) / kThreads;
    blocks = blocks < 1 ? 1 : (blocks > 65535 ? 65535 : blocks);
    pool3d<<<static_cast<unsigned>(blocks), kThreads>>>(input, output, static_cast<int>(H), h_out, static_cast<int>(W), w_out, static_cast<int>(D), d_out, kernel_size, stride, padding, dilation);
}

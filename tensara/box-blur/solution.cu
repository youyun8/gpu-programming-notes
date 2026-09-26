// Box Blur (Tensara)
// https://tensara.org/problems/box-blur
//
// Mean over the in-bounds part of a k x k window around every pixel.
// A box filter is separable: sum along rows, then along columns. Each pass is
// O(k) per pixel (instead of O(k^2)), and the divisor is simply the product
// of the in-bounds window extents in each direction.
#include <cuda_runtime.h>

constexpr int kThreads = 256;

__global__ void rowSums(const float* __restrict__ in, float* __restrict__ tmp, int h, int w, int half) {
    const size_t total = static_cast<size_t>(h) * w;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        // Horizontal pass: sum of the in-bounds part of the window along this row.
        const int x = static_cast<int>(idx % w);
        const float* row = in + (idx - x);
        const int lo = max(x - half, 0), hi = min(x + half, w - 1);
        float s = 0.0f;
        for (int j = lo; j <= hi; ++j) s += row[j];
        tmp[idx] = s;
    }
}

__global__ void colSums(const float* __restrict__ tmp, float* __restrict__ out, int h, int w, int half) {
    const size_t total = static_cast<size_t>(h) * w;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        // Vertical pass over the row sums; neighbouring threads read neighbouring columns (coalesced).
        const int x = static_cast<int>(idx % w);
        const int y = static_cast<int>(idx / w);
        const int lo = max(y - half, 0), hi = min(y + half, h - 1);
        float s = 0.0f;
        for (int i = lo; i <= hi; ++i) s += tmp[static_cast<size_t>(i) * w + x];
        // Divide by the number of valid pixels: (valid rows) * (valid columns) of the clipped window.
        const int cols = min(x + half, w - 1) - max(x - half, 0) + 1;
        out[idx] = s / static_cast<float>((hi - lo + 1) * cols);
    }
}

// input_image, output_image are device pointers
extern "C" void solution(const float* input_image, int kernel_size, float* output_image, size_t height, size_t width) {
    const int h = static_cast<int>(height), w = static_cast<int>(width);
    // Temporary image for the row sums; the box filter is separable (O(k) per pixel, not O(k^2)).
    float* tmp = nullptr;
    cudaMalloc(&tmp, height * width * sizeof(float));
    size_t blocks = (height * width + kThreads - 1) / kThreads;
    blocks = blocks > 65535 ? 65535 : blocks;
    rowSums<<<static_cast<unsigned>(blocks), kThreads>>>(input_image, tmp, h, w, kernel_size / 2);
    colSums<<<static_cast<unsigned>(blocks), kThreads>>>(tmp, output_image, h, w, kernel_size / 2);
    // Wait for both passes before freeing the temporary.
    cudaDeviceSynchronize();
    cudaFree(tmp);
}

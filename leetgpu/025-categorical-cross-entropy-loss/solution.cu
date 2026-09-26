// Categorical Cross-Entropy Loss (LeetGPU)
// https://leetgpu.com/challenges/categorical-cross-entropy-loss
//
// loss = mean_j( logsumexp(z_j) - z_j[y_j] ).
// One warp per sample computes a numerically stable logsumexp with an online
// (max, sum) pair per lane, merged with shuffles. Per-sample losses are summed
// in fp64 into per-block partials; a final block divides by N.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kWarpsPerBlock = 8;
constexpr int kMaxBlocks = 1024;

__device__ double g_partials[kMaxBlocks];

__global__ void sampleLosses(const float* logits, const int* labels, int n, int c) {
    __shared__ double warp_loss[kWarpsPerBlock];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    double block_total = 0.0;
    for (int row = blockIdx.x * kWarpsPerBlock + warp; row < n; row += gridDim.x * kWarpsPerBlock) {
        const float* z = logits + static_cast<size_t>(row) * c;
        float m = -FLT_MAX;
        float s = 0.0f;
        for (int j = lane; j < c; j += 32) {
            const float v = z[j];
            const float new_m = fmaxf(m, v);
            s = s * expf(m - new_m) + expf(v - new_m);
            m = new_m;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float om = __shfl_xor_sync(0xffffffffu, m, offset);
            const float os = __shfl_xor_sync(0xffffffffu, s, offset);
            const float new_m = fmaxf(m, om);
            s = s * expf(m - new_m) + os * expf(om - new_m);
            m = new_m;
        }
        if (lane == 0) block_total += static_cast<double>(m + logf(s) - z[labels[row]]);
    }
    if (lane == 0) warp_loss[warp] = block_total;
    __syncthreads();
    if (threadIdx.x == 0) {
        double t = 0.0;
        for (int w = 0; w < kWarpsPerBlock; ++w) t += warp_loss[w];
        g_partials[blockIdx.x] = t;
    }
}

__global__ void finalMean(float* loss, int num_partials, int n) {
    if (threadIdx.x == 0) {
        double t = 0.0;
        for (int i = 0; i < num_partials; ++i) t += g_partials[i];
        loss[0] = static_cast<float>(t / n);
    }
}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    int num_blocks = (N + kWarpsPerBlock - 1) / kWarpsPerBlock;
    num_blocks = num_blocks > kMaxBlocks ? kMaxBlocks : num_blocks;
    sampleLosses<<<num_blocks, kWarpsPerBlock * 32>>>(logits, true_labels, N, C);
    finalMean<<<1, 32>>>(loss, num_blocks, N);
    cudaDeviceSynchronize();
}

// 1D Convolution (Tensara)
// https://tensara.org/problems/conv-1d
//
// "Same" 1D convolution (cross-correlation), odd kernel of up to ~8K taps,
// zero padding K/2 on both sides: C[i] = sum_j A[i + j - K/2] * B[j].
// Each block produces 1024 outputs (4 per thread, strided by 256 so that
// neighbouring threads read neighbouring shared words). Because K can be
// large, the kernel is processed in chunks of 2048 taps: per chunk the block
// stages the taps and the matching input window (1024 + 2047 floats) in
// shared memory, so every input element is read from DRAM ~once per chunk.
#include <cuda_runtime.h>

constexpr int kThreads = 256;
constexpr int kOutPerThread = 4;
constexpr int kOutPerBlock = kThreads * kOutPerThread;
constexpr int kTapChunk = 2048;

__global__ void conv1dSame(const float* __restrict__ in, const float* __restrict__ w, float* __restrict__ out, long long n, int k) {
    __shared__ float s_w[kTapChunk];
    __shared__ float s_in[kOutPerBlock + kTapChunk - 1];
    const long long base = static_cast<long long>(blockIdx.x) * kOutPerBlock;
    const int half = k / 2;
    float acc[kOutPerThread] = {};
    for (int t0 = 0; t0 < k; t0 += kTapChunk) {
        const int taps = min(kTapChunk, k - t0);
        const int window = kOutPerBlock + taps - 1;
        __syncthreads();
        for (int i = threadIdx.x; i < taps; i += kThreads) s_w[i] = w[t0 + i];
        for (int i = threadIdx.x; i < window; i += kThreads) {
            const long long g = base + i + t0 - half;
            s_in[i] = (g >= 0 && g < n) ? in[g] : 0.0f;
        }
        __syncthreads();
        for (int j = 0; j < taps; ++j) {
            const float wj = s_w[j];
#pragma unroll
            for (int r = 0; r < kOutPerThread; ++r) acc[r] = fmaf(s_in[threadIdx.x + r * kThreads + j], wj, acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < kOutPerThread; ++r) {
        const long long o = base + threadIdx.x + r * kThreads;
        if (o < n) out[o] = acc[r];
    }
}

// A, B, C are device pointers
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    const unsigned blocks = static_cast<unsigned>((N + kOutPerBlock - 1) / kOutPerBlock);
    conv1dSame<<<blocks, kThreads>>>(A, B, C, static_cast<long long>(N), static_cast<int>(K));
}

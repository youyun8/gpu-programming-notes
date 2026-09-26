// Matmul + Sigmoid + Sum (Tensara)
// https://tensara.org/problems/matmul-sigmoid-sum
//
// sum(sigmoid(A B)) as one scalar. The M x N product is never stored: the
// register-blocked SGEMM epilogue applies the sigmoid, every block reduces
// its 64 x 64 tile and adds one fp64 partial with atomicAdd; a one-thread
// kernel converts the total to float.
#include <cuda_runtime.h>

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreads = 256;

__device__ double g_sum;

__global__ void __launch_bounds__(kThreads)
sigmoidSumGemm(const float* a, const float* b, int rows, int inner, int cols) {
    __shared__ float a_tile[kTileK][kTileM + 4];
    __shared__ float b_tile[kTileK][kTileN + 4];
    __shared__ float warp_sums[kThreads / 32];
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int row0 = blockIdx.y * kTileM, col0 = blockIdx.x * kTileN;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < inner; k0 += kTileK) {
        for (int i = tid; i < kTileM * kTileK; i += kThreads) {
            const int r = i / kTileK, kk = i % kTileK;
            a_tile[kk][r] = (row0 + r < rows && k0 + kk < inner) ? a[static_cast<size_t>(row0 + r) * inner + k0 + kk] : 0.0f;
        }
        for (int i = tid; i < kTileK * kTileN; i += kThreads) {
            const int kk = i / kTileN, cc = i % kTileN;
            b_tile[kk][cc] = (k0 + kk < inner && col0 + cc < cols) ? b[static_cast<size_t>(k0 + kk) * cols + col0 + cc] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            float af[4], bf[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) af[i] = a_tile[kk][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) bf[j] = b_tile[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }
    float local = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const int r = row0 + ty + 16 * i;
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + tx + 16 * j;
            if (r < rows && col < cols) local += 1.0f / (1.0f + expf(-acc[i][j]));
        }
    }
    for (int o = 16; o > 0; o >>= 1) local += __shfl_xor_sync(0xffffffffu, local, o);
    if (threadIdx.x % 32 == 0) warp_sums[threadIdx.x / 32] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        double t = 0.0;
        for (int w = 0; w < kThreads / 32; ++w) t += warp_sums[w];
        atomicAdd(&g_sum, t);
    }
}

__global__ void resetSum() { g_sum = 0.0; }
__global__ void writeSum(float* out) { out[0] = static_cast<float>(g_sum); }

// A, B, output are device pointers
extern "C" void solution(const float* A, const float* B, float* output, size_t M, size_t N, size_t K) {
    const dim3 grid(static_cast<unsigned>((N + kTileN - 1) / kTileN), static_cast<unsigned>((M + kTileM - 1) / kTileM));
    resetSum<<<1, 1>>>();
    sigmoidSumGemm<<<grid, kThreads>>>(A, B, static_cast<int>(M), static_cast<int>(K), static_cast<int>(N));
    writeSum<<<1, 1>>>(output);
}

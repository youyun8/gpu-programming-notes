// 2D FFT (LeetGPU)
// https://leetgpu.com/challenges/2d-fft
//
// Row-column decomposition with transposes so that every pass reads rows:
//   rows(M x N) -> transpose -> rows(N x M) -> transpose back.
// Row transforms run one block per row in shared memory:
//   - power-of-two length: iterative radix-2 Cooley-Tukey (bit-reversal
//     permutation on load, log2(n) butterfly stages);
//   - other lengths: direct O(n^2) DFT (only small odd sizes are expected).
// Twiddles use sincospif(2k/n) with k reduced mod n in integers, so the
// angles are exact even for n = 4096.
#include <cuda_runtime.h>

constexpr int kThreads = 512;
constexpr int kMaxLen = 4096;
constexpr int kTile = 32;

__device__ __forceinline__ float2 cmul(float2 a, float2 b) { return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

// exp(-2*pi*i*k/n)
__device__ __forceinline__ float2 twiddle(long long k, int n) {
    float s, c;
    sincospif(-2.0f * static_cast<float>(k % n) / static_cast<float>(n), &s, &c);
    return make_float2(c, s);
}

__global__ void fftRows(const float2* in, float2* out, int n) {
    extern __shared__ float2 s_row[];
    const float2* src = in + static_cast<size_t>(blockIdx.x) * n;
    float2* dst = out + static_cast<size_t>(blockIdx.x) * n;
    const bool pow2 = (n & (n - 1)) == 0;
    if (pow2) {
        int log_n = 0;
        while ((1 << log_n) < n) ++log_n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const unsigned rev = log_n ? (__brev(static_cast<unsigned>(i)) >> (32 - log_n)) : 0u;
            s_row[rev] = src[i];
        }
        __syncthreads();
        for (int half = 1; half < n; half <<= 1) {
            for (int t = threadIdx.x; t < n / 2; t += blockDim.x) {
                const int group = t / half;
                const int pos = t % half;
                const int i0 = group * 2 * half + pos;
                const int i1 = i0 + half;
                const float2 w = twiddle(static_cast<long long>(pos) * (n / (2 * half)), n);
                const float2 u = s_row[i0];
                const float2 v = cmul(s_row[i1], w);
                s_row[i0] = make_float2(u.x + v.x, u.y + v.y);
                s_row[i1] = make_float2(u.x - v.x, u.y - v.y);
            }
            __syncthreads();
        }
        for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = s_row[i];
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) s_row[i] = src[i];
        __syncthreads();
        for (int k = threadIdx.x; k < n; k += blockDim.x) {
            float2 acc = make_float2(0.0f, 0.0f);
            for (int j = 0; j < n; ++j) {
                const float2 p = cmul(s_row[j], twiddle(static_cast<long long>(j) * k, n));
                acc.x += p.x;
                acc.y += p.y;
            }
            dst[k] = acc;
        }
    }
}

// out (cols x rows) = in (rows x cols)^T for complex elements.
__global__ void transposeComplex(const float2* in, float2* out, int rows, int cols) {
    __shared__ float2 tile[kTile][kTile + 1];
    int x = blockIdx.x * kTile + threadIdx.x;
    int y = blockIdx.y * kTile + threadIdx.y;
    for (int j = 0; j < kTile; j += blockDim.y)
        if (x < cols && y + j < rows) tile[threadIdx.y + j][threadIdx.x] = in[static_cast<size_t>(y + j) * cols + x];
    __syncthreads();
    x = blockIdx.y * kTile + threadIdx.x;
    y = blockIdx.x * kTile + threadIdx.y;
    for (int j = 0; j < kTile; j += blockDim.y)
        if (x < rows && y + j < cols) out[static_cast<size_t>(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
}

// signal, spectrum are device pointers
extern "C" void solve(const float* signal, float* spectrum, int M, int N) {
    const float2* in = reinterpret_cast<const float2*>(signal);
    float2* out = reinterpret_cast<float2*>(spectrum);
    float2* tmp = nullptr;
    cudaMalloc(&tmp, static_cast<size_t>(M) * N * sizeof(float2));
    const int max_len = M > N ? M : N;
    cudaFuncSetAttribute(fftRows, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxLen * sizeof(float2));
    (void)max_len;

    fftRows<<<M, kThreads, N * sizeof(float2)>>>(in, out, N);                                  // rows of length N
    transposeComplex<<<dim3((N + kTile - 1) / kTile, (M + kTile - 1) / kTile), dim3(kTile, 8)>>>(out, tmp, M, N);
    fftRows<<<N, kThreads, M * sizeof(float2)>>>(tmp, tmp, M);                                 // columns (now rows) of length M
    transposeComplex<<<dim3((M + kTile - 1) / kTile, (N + kTile - 1) / kTile), dim3(kTile, 8)>>>(tmp, out, N, M);
    cudaDeviceSynchronize();
    cudaFree(tmp);
}

// Fast Fourier Transform (LeetGPU)
// https://leetgpu.com/challenges/fast-fourier-transform
//
// Any N <= 262,144, entirely on the GPU:
//   - power-of-two N: Stockham radix-2 FFT, log2(N) passes in global memory
//     (self-sorting, so no bit-reversal permutation is needed);
//   - otherwise Bluestein's chirp-z algorithm: with w_n = exp(-i pi n^2 / N),
//       X_k = w_k * sum_n (x_n w_n) conj(w_{k-n}),
//     a linear convolution evaluated with power-of-two FFTs of size L >= 2N - 1.
// Chirp phases reduce n^2 mod 2N in 64-bit integers before sincospif, so they
// stay exact for large n.
#include <cuda_runtime.h>

constexpr int kThreads = 256;

__device__ __forceinline__ float2 cmul(float2 a, float2 b) { return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

// One Stockham radix-2 pass; ns = size of the sub-transforms already done.
// sign = -1 forward, +1 inverse (unnormalized).
__global__ void stockhamPass(const float2* in, float2* out, int n, int ns, float sign) {
    const int half = n / 2;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < half; j += gridDim.x * blockDim.x) {
        const int k = j % ns;
        float s, c;
        sincospif(sign * static_cast<float>(k) / static_cast<float>(ns), &s, &c);
        const float2 v0 = in[j];
        const float2 v1 = cmul(in[j + half], make_float2(c, s));
        const int dst = (j / ns) * ns * 2 + k;
        out[dst] = make_float2(v0.x + v1.x, v0.y + v1.y);
        out[dst + ns] = make_float2(v0.x - v1.x, v0.y - v1.y);
    }
}

static int gridFor(int work) {
    const int b = (work + kThreads - 1) / kThreads;
    return b < 1 ? 1 : (b > 4096 ? 4096 : b);
}

// FFT of length n (power of two) from a into a (b is scratch of the same size).
static void fftPow2(float2* a, float2* b, int n, float sign) {
    float2* src = a;
    float2* dst = b;
    for (int ns = 1; ns < n; ns *= 2) {
        stockhamPass<<<gridFor(n / 2), kThreads>>>(src, dst, n, ns, sign);
        float2* t = src;
        src = dst;
        dst = t;
    }
    if (src != a) cudaMemcpyAsync(a, src, n * sizeof(float2), cudaMemcpyDeviceToDevice);
}

__device__ __forceinline__ float2 chirp(long long idx, int n) {  // exp(-i pi idx^2 / n)
    const long long m = (idx * idx) % (2LL * n);
    float s, c;
    sincospif(-static_cast<float>(m) / static_cast<float>(n), &s, &c);
    return make_float2(c, s);
}

__global__ void bluesteinPrep(const float2* x, float2* a, float2* b, int n, int len) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x) {
        a[i] = i < n ? cmul(x[i], chirp(i, n)) : make_float2(0.0f, 0.0f);
        // b = conj(chirp) at offsets 0..n-1 and their mirror images len-(n-1)..len-1.
        float2 bv = make_float2(0.0f, 0.0f);
        if (i < n) {
            const float2 w = chirp(i, n);
            bv = make_float2(w.x, -w.y);
        } else if (i > len - n) {
            const float2 w = chirp(len - i, n);
            bv = make_float2(w.x, -w.y);
        }
        b[i] = bv;
    }
}

__global__ void pointwiseMul(float2* a, const float2* b, int len) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x) a[i] = cmul(a[i], b[i]);
}

__global__ void bluesteinFinish(const float2* conv, float2* out, int n, int len) {
    const float inv = 1.0f / static_cast<float>(len);
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x) {
        const float2 c = conv[k];
        out[k] = cmul(make_float2(c.x * inv, c.y * inv), chirp(k, n));
    }
}

// signal, spectrum are device pointers
extern "C" void solve(const float* signal, float* spectrum, int N) {
    const float2* x = reinterpret_cast<const float2*>(signal);
    float2* out = reinterpret_cast<float2*>(spectrum);
    if ((N & (N - 1)) == 0) {
        float2* scratch = nullptr;
        cudaMalloc(&scratch, N * sizeof(float2));
        cudaMemcpy(out, x, N * sizeof(float2), cudaMemcpyDeviceToDevice);
        fftPow2(out, scratch, N, -1.0f);
        cudaDeviceSynchronize();
        cudaFree(scratch);
        return;
    }
    int len = 1;
    while (len < 2 * N - 1) len <<= 1;
    float2* buf = nullptr;
    cudaMalloc(&buf, 3 * static_cast<size_t>(len) * sizeof(float2));
    float2* a = buf;
    float2* b = buf + len;
    float2* scratch = buf + 2 * static_cast<size_t>(len);
    bluesteinPrep<<<gridFor(len), kThreads>>>(x, a, b, N, len);
    fftPow2(a, scratch, len, -1.0f);
    fftPow2(b, scratch, len, -1.0f);
    pointwiseMul<<<gridFor(len), kThreads>>>(a, b, len);
    fftPow2(a, scratch, len, 1.0f);  // inverse (unnormalized); 1/len applied below
    bluesteinFinish<<<gridFor(N), kThreads>>>(a, out, N, len);
    cudaDeviceSynchronize();
    cudaFree(buf);
}

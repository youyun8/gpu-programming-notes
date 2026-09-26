// Logistic Regression (LeetGPU)
// https://leetgpu.com/challenges/logistic-regression
//
// Newton-Raphson / IRLS, the same iteration as the reference, in fp64:
//   p = sigmoid(X beta), W = max(p (1 - p), 1e-8)
//   g = X^T (p - y) + lambda beta,   H = X^T diag(W) X + lambda I
//   beta -= H^-1 g   until ||step|| < 1e-8 (at most 1000 iterations)
// Per iteration: one kernel for p/W/residuals, a tiled weighted-Gram kernel
// for H, one for g, and a single-block Cholesky solve that also updates beta
// and reports ||step||^2 (the only value copied back to the host).
#include <cuda_runtime.h>

constexpr int kTile = 16;
constexpr int kSolveThreads = 1024;
constexpr int kMaxIter = 1000;
constexpr double kTol = 1e-8;
constexpr double kL2 = 1e-6;

__global__ void sampleTerms(const float* x, const float* y, const double* beta, double* w, double* r, int n, int f) {
    const int lane = threadIdx.x % 32;
    const int s = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (s >= n) return;
    double z = 0.0;
    for (int j = lane; j < f; j += 32) z += static_cast<double>(x[static_cast<size_t>(s) * f + j]) * beta[j];
    for (int offset = 16; offset > 0; offset >>= 1) z += __shfl_down_sync(0xffffffffu, z, offset);
    if (lane == 0) {
        const double p = 1.0 / (1.0 + exp(-z));
        w[s] = fmax(p * (1.0 - p), 1e-8);
        r[s] = p - y[s];
    }
}

__global__ void weightedGram(const float* x, const double* w, double* h, int n, int f) {
    __shared__ double xi[kTile][kTile + 1];
    __shared__ double xj[kTile][kTile + 1];
    const int i0 = blockIdx.y * kTile;
    const int j0 = blockIdx.x * kTile;
    double acc = 0.0;
    for (int s0 = 0; s0 < n; s0 += kTile) {
        const int s = s0 + threadIdx.y;
        const double ws = s < n ? w[s] : 0.0;
        xi[threadIdx.y][threadIdx.x] = (s < n && i0 + threadIdx.x < f) ? ws * x[static_cast<size_t>(s) * f + i0 + threadIdx.x] : 0.0;
        xj[threadIdx.y][threadIdx.x] = (s < n && j0 + threadIdx.x < f) ? x[static_cast<size_t>(s) * f + j0 + threadIdx.x] : 0.0;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kTile; ++k) acc += xi[k][threadIdx.y] * xj[k][threadIdx.x];
        __syncthreads();
    }
    const int i = i0 + threadIdx.y;
    const int j = j0 + threadIdx.x;
    if (i < f && j < f) h[static_cast<size_t>(i) * f + j] = acc + (i == j ? kL2 : 0.0);
}

__global__ void gradient(const float* x, const double* r, const double* beta, double* g, int n, int f) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= f) return;
    double acc = 0.0;
    for (int s = 0; s < n; ++s) acc += static_cast<double>(x[static_cast<size_t>(s) * f + j]) * r[s];
    g[j] = acc + kL2 * beta[j];
}

__device__ double blockSum(double v, double* scratch) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = v;
    __syncthreads();
    double total = 0.0;
    for (int wi = 0; wi < static_cast<int>(blockDim.x / 32); ++wi) total += scratch[wi];
    __syncthreads();
    return total;
}

// Solves H delta = g (Cholesky, in place), beta -= delta, *step_sq = ||delta||^2.
__global__ void newtonStep(double* a, const double* g, double* z, double* beta, double* step_sq, int f) {
    __shared__ double scratch[32];
    __shared__ double pivot;
    for (int k = 0; k < f; ++k) {
        if (threadIdx.x == 0) {
            pivot = sqrt(a[static_cast<size_t>(k) * f + k]);
            a[static_cast<size_t>(k) * f + k] = pivot;
        }
        __syncthreads();
        for (int i = k + 1 + threadIdx.x; i < f; i += blockDim.x) a[static_cast<size_t>(i) * f + k] /= pivot;
        __syncthreads();
        const int m = f - k - 1;
        for (long long t = threadIdx.x; t < static_cast<long long>(m) * m; t += blockDim.x) {
            const int i = k + 1 + static_cast<int>(t / m);
            const int j = k + 1 + static_cast<int>(t % m);
            if (j <= i) a[static_cast<size_t>(i) * f + j] -= a[static_cast<size_t>(i) * f + k] * a[static_cast<size_t>(j) * f + k];
        }
        __syncthreads();
    }
    for (int i = 0; i < f; ++i) {
        double partial = 0.0;
        for (int j = threadIdx.x; j < i; j += blockDim.x) partial += a[static_cast<size_t>(i) * f + j] * z[j];
        const double s = blockSum(partial, scratch);
        if (threadIdx.x == 0) z[i] = (g[i] - s) / a[static_cast<size_t>(i) * f + i];
        __syncthreads();
    }
    for (int i = f - 1; i >= 0; --i) {
        double partial = 0.0;
        for (int j = i + 1 + threadIdx.x; j < f; j += blockDim.x) partial += a[static_cast<size_t>(j) * f + i] * z[j];
        const double s = blockSum(partial, scratch);
        if (threadIdx.x == 0) z[i] = (z[i] - s) / a[static_cast<size_t>(i) * f + i];
        __syncthreads();
    }
    double partial = 0.0;
    for (int i = threadIdx.x; i < f; i += blockDim.x) {
        beta[i] -= z[i];
        partial += z[i] * z[i];
    }
    const double total = blockSum(partial, scratch);
    if (threadIdx.x == 0) *step_sq = total;
}

__global__ void writeBeta(const double* beta_d, float* beta, int f) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < f) beta[i] = static_cast<float>(beta_d[i]);
}

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    const int n = n_samples;
    const int f = n_features;
    double* buf = nullptr;
    const size_t count = static_cast<size_t>(f) * f + 3 * static_cast<size_t>(f) + 2 * static_cast<size_t>(n) + 1;
    cudaMalloc(&buf, count * sizeof(double));
    cudaMemset(buf, 0, count * sizeof(double));
    double* h = buf;
    double* g = h + static_cast<size_t>(f) * f;
    double* z = g + f;
    double* beta_d = z + f;
    double* w = beta_d + f;
    double* r = w + n;
    double* step_sq = r + n;

    const dim3 gram_grid((f + kTile - 1) / kTile, (f + kTile - 1) / kTile);
    for (int it = 0; it < kMaxIter; ++it) {
        sampleTerms<<<(n + 7) / 8, 256>>>(X, y, beta_d, w, r, n, f);
        weightedGram<<<gram_grid, dim3(kTile, kTile)>>>(X, w, h, n, f);
        gradient<<<(f + 255) / 256, 256>>>(X, r, beta_d, g, n, f);
        newtonStep<<<1, kSolveThreads>>>(h, g, z, beta_d, step_sq, f);
        double host_step_sq = 0.0;
        cudaMemcpy(&host_step_sq, step_sq, sizeof(double), cudaMemcpyDeviceToHost);
        if (host_step_sq < kTol * kTol) break;
    }
    writeBeta<<<(f + 255) / 256, 256>>>(beta_d, beta, f);
    cudaDeviceSynchronize();
    cudaFree(buf);
}

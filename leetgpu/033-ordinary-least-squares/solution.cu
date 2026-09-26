// Ordinary Least Squares (LeetGPU)
// https://leetgpu.com/challenges/ordinary-least-squares
//
// beta = (X^T X)^-1 X^T y via the normal equations:
//   1. gramTiled: G = X^T X (F x F) as a tiled "A^T A" GEMM with fp64 accumulation;
//      X^T y as one thread per feature (coalesced over features).
//   2. choleskySolve: one block factors G = L L^T in fp64 (right-looking,
//      in place) and does the forward / backward substitutions.
// fp64 matters here: X^T X squares the condition number of X.
#include <cuda_runtime.h>

constexpr int kTile = 16;
constexpr int kSolveThreads = 1024;

__global__ void gramTiled(const float* x, double* gram, int n, int f) {
    __shared__ float xi[kTile][kTile + 1];
    __shared__ float xj[kTile][kTile + 1];
    const int i0 = blockIdx.y * kTile;
    const int j0 = blockIdx.x * kTile;
    const int ti = threadIdx.y;
    const int tj = threadIdx.x;
    double acc = 0.0;
    for (int s0 = 0; s0 < n; s0 += kTile) {
        const int s = s0 + threadIdx.y;
        xi[threadIdx.y][threadIdx.x] = (s < n && i0 + threadIdx.x < f) ? x[static_cast<size_t>(s) * f + i0 + threadIdx.x] : 0.0f;
        xj[threadIdx.y][threadIdx.x] = (s < n && j0 + threadIdx.x < f) ? x[static_cast<size_t>(s) * f + j0 + threadIdx.x] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kTile; ++k) acc += static_cast<double>(xi[k][ti]) * xj[k][tj];
        __syncthreads();
    }
    if (i0 + ti < f && j0 + tj < f) gram[static_cast<size_t>(i0 + ti) * f + j0 + tj] = acc;
}

__global__ void xtY(const float* x, const float* y, double* rhs, int n, int f) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= f) return;
    double acc = 0.0;
    for (int s = 0; s < n; ++s) acc += static_cast<double>(x[static_cast<size_t>(s) * f + j]) * y[s];
    rhs[j] = acc;
}

__device__ double blockSum(double v, double* scratch) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = v;
    __syncthreads();
    double total = 0.0;
    for (int w = 0; w < static_cast<int>(blockDim.x / 32); ++w) total += scratch[w];
    __syncthreads();
    return total;
}

// In-place Cholesky of the F x F matrix a (lower triangle), then solves
// L z = b and L^T beta = z. Single block.
__global__ void choleskySolve(double* a, const double* b, float* beta, double* z, int f) {
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
        // Trailing update of the lower triangle: a[i][j] -= L[i][k] * L[j][k], j <= i.
        const int m = f - k - 1;
        for (long long t = threadIdx.x; t < static_cast<long long>(m) * m; t += blockDim.x) {
            const int i = k + 1 + static_cast<int>(t / m);
            const int j = k + 1 + static_cast<int>(t % m);
            if (j <= i) a[static_cast<size_t>(i) * f + j] -= a[static_cast<size_t>(i) * f + k] * a[static_cast<size_t>(j) * f + k];
        }
        __syncthreads();
    }
    for (int i = 0; i < f; ++i) {  // forward: L z = b
        double partial = 0.0;
        for (int j = threadIdx.x; j < i; j += blockDim.x) partial += a[static_cast<size_t>(i) * f + j] * z[j];
        const double s = blockSum(partial, scratch);
        if (threadIdx.x == 0) z[i] = (b[i] - s) / a[static_cast<size_t>(i) * f + i];
        __syncthreads();
    }
    for (int i = f - 1; i >= 0; --i) {  // backward: L^T beta = z (reuse z in place)
        double partial = 0.0;
        for (int j = i + 1 + threadIdx.x; j < f; j += blockDim.x) partial += a[static_cast<size_t>(j) * f + i] * z[j];
        const double s = blockSum(partial, scratch);
        if (threadIdx.x == 0) z[i] = (z[i] - s) / a[static_cast<size_t>(i) * f + i];
        __syncthreads();
    }
    for (int i = threadIdx.x; i < f; i += blockDim.x) beta[i] = static_cast<float>(z[i]);
}

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    const int f = n_features;
    double* buf = nullptr;
    cudaMalloc(&buf, (static_cast<size_t>(f) * f + 2 * f) * sizeof(double));
    double* gram = buf;
    double* rhs = gram + static_cast<size_t>(f) * f;
    double* z = rhs + f;
    const dim3 grid((f + kTile - 1) / kTile, (f + kTile - 1) / kTile);
    gramTiled<<<grid, dim3(kTile, kTile)>>>(X, gram, n_samples, f);
    xtY<<<(f + 255) / 256, 256>>>(X, y, rhs, n_samples, f);
    choleskySolve<<<1, kSolveThreads>>>(gram, rhs, beta, z, f);
    cudaDeviceSynchronize();
    cudaFree(buf);
}

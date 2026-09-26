// Linear Attention (LeetGPU)
// https://leetgpu.com/challenges/linear-self-attention
//
// out_i = phi(Q_i) S / (phi(Q_i) . z),  S = phi(K)^T V (d x d),  z = sum_j phi(K_j),
// phi(x) = elu(x) + 1. Associativity turns O(M^2 d) attention into O(M d^2):
//   1. kvState: one thread per entry of S (and of z) reduces over all M rows in
//      fp64; neighbouring threads read neighbouring V columns (coalesced) while
//      phi(K[m][i]) is a broadcast;
//   2. applyState: one block per query row; phi(Q_i) and its dot with z are
//      computed once in shared memory, then thread j forms (phi(Q_i) S)[j].
#include <cuda_runtime.h>

constexpr int kMaxDim = 128;

// Feature map phi(x) = elu(x) + 1: x + 1 for x > 0, e^x otherwise (always positive).
__device__ __forceinline__ float phi(float x) { return x > 0.0f ? x + 1.0f : expf(x); }

__global__ void kvState(const float* k, const float* v, float* s, float* z, int m, int d) {
    // Threads 0 .. d*d-1: S = phi(K)^T V (d x d). Threads d*d .. d*d+d-1: z = column sums of phi(K).
    // Both accumulate over all M rows in double.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d * d) {
        const int i = idx / d, j = idx % d;
        double acc = 0.0;
        for (int r = 0; r < m; ++r) acc += static_cast<double>(phi(k[static_cast<size_t>(r) * d + i])) * v[static_cast<size_t>(r) * d + j];
        s[idx] = static_cast<float>(acc);
    } else if (idx < d * d + d) {
        const int i = idx - d * d;
        double acc = 0.0;
        for (int r = 0; r < m; ++r) acc += phi(k[static_cast<size_t>(r) * d + i]);
        z[i] = static_cast<float>(acc);
    }
}

__global__ void applyState(const float* q, const float* s, const float* z, float* out, int d) {
    // One block per query row: phi(q), then denominator phi(q) . z ...
    __shared__ float phi_q[kMaxDim];
    __shared__ float denom;
    const size_t row = blockIdx.x;
    for (int i = threadIdx.x; i < d; i += blockDim.x) phi_q[i] = phi(q[row * d + i]);
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < d; ++i) t = fmaf(phi_q[i], z[i], t);
        denom = t;
    }
    __syncthreads();
    // ... and output column j = (phi(q) . S[:, j]) / denominator.
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float num = 0.0f;
        for (int i = 0; i < d; ++i) num = fmaf(phi_q[i], s[i * d + j], num);
        out[row * d + j] = num / denom;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    // The d x d state and the normalizer replace the M x M attention matrix.
    float* state = nullptr;
    cudaMalloc(&state, (static_cast<size_t>(d) * d + d) * sizeof(float));
    kvState<<<(d * d + d + 255) / 256, 256>>>(K, V, state, state + d * d, M, d);
    applyState<<<M, kMaxDim>>>(Q, state, state + d * d, output, d);
    cudaDeviceSynchronize();
    cudaFree(state);
}

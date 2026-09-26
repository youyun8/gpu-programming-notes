// Single-Source Shortest Path (Tensara)
// https://tensara.org/problems/shortest-path
//
// Dense adjacency matrix, positive weights (0 = no edge). Bellman-Ford style
// relaxation, parallel over destination vertices: thread v computes
// min(dist[v], min_u dist[u] + w[u][v]) reading column v, which is coalesced
// across the warp for each u. Iterations stop as soon as a sweep changes
// nothing (usually after ~graph-diameter sweeps, far fewer than N - 1).
// Unreachable vertices are reported as -1.
#include <cuda_runtime.h>

constexpr int kThreads = 256;

// Set by any thread whose distance improved in the current sweep.
__device__ int g_changed;

__global__ void initDist(float* dist, int n, int source) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    // Distances start at +inf except the source.
    if (v < n) dist[v] = v == source ? 0.0f : __int_as_float(0x7f800000);
}

__global__ void relax(const float* __restrict__ adj, const float* __restrict__ dist, float* __restrict__ next, int n) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    // Bellman-Ford sweep, one thread per destination v: relax through every u. Reading
    // column v means the warp reads 32 consecutive floats of row u at each step (coalesced).
    float best = dist[v];
    for (int u = 0; u < n; ++u) {
        const float w = adj[static_cast<size_t>(u) * n + v];
        const float du = dist[u];
        if (w > 0.0f && du + w < best) best = du + w;
    }
    // Double-buffered (Jacobi) update, and report any change.
    next[v] = best;
    if (best != dist[v]) g_changed = 1;
}

__global__ void finish(const float* dist, float* out, int n) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    // Unreachable vertices are reported as -1.
    if (v < n) out[v] = isinf(dist[v]) ? -1.0f : dist[v];
}

// d_adj_matrix, d_distances are device pointers
extern "C" void solution(const float* d_adj_matrix, int source, float* d_distances, size_t n) {
    const int N = static_cast<int>(n);
    const int blocks = (N + kThreads - 1) / kThreads;
    // Two distance buffers, swapped after every sweep.
    float* buf = nullptr;
    cudaMalloc(&buf, 2 * n * sizeof(float));
    float* cur = buf;
    float* next = buf + n;
    initDist<<<blocks, kThreads>>>(cur, N, source);
    // At most N - 1 sweeps; stop as soon as a sweep changes nothing (reading the flag
    // synchronizes with the device each iteration).
    for (int it = 0; it < N - 1; ++it) {
        const int zero = 0;
        cudaMemcpyToSymbol(g_changed, &zero, sizeof(int));
        relax<<<blocks, kThreads>>>(d_adj_matrix, cur, next, N);
        int changed = 0;
        cudaMemcpyFromSymbol(&changed, g_changed, sizeof(int));
        float* t = cur;
        cur = next;
        next = t;
        if (!changed) break;
    }
    finish<<<blocks, kThreads>>>(cur, d_distances, N);
    cudaDeviceSynchronize();
    cudaFree(buf);
}

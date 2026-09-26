// Minimum Spanning Tree weight (Tensara)
// https://tensara.org/problems/min-spanning-tree
//
// Prim's algorithm on a dense adjacency matrix (0 = no edge) in one block:
// each of the N - 1 steps is a block-wide arg-min over the vertices not yet in
// the tree (ties -> lowest index), then every thread relaxes its vertices'
// best edge with the new vertex's row (coalesced). O(N^2) work with only
// block barriers, no kernel launches per step. The total is accumulated in
// fp64; a disconnected graph yields +inf.
#include <cuda_runtime.h>

constexpr int kThreads = 1024;

__global__ void prim(const float* adj, float* best, unsigned char* in_tree, float* result, int n) {
    __shared__ float s_val[kThreads / 32];
    __shared__ int s_idx[kThreads / 32];
    __shared__ int s_pick;
    __shared__ float s_pick_val;
    const float inf = __int_as_float(0x7f800000);
    for (int v = threadIdx.x; v < n; v += kThreads) {
        const float w = adj[v];
        best[v] = v == 0 ? 0.0f : (w == 0.0f ? inf : w);
        in_tree[v] = v == 0;
    }
    __syncthreads();
    double total = 0.0;
    for (int step = 0; step < n - 1; ++step) {
        float val = inf;
        int idx = 0x7fffffff;
        for (int v = threadIdx.x; v < n; v += kThreads) {
            if (!in_tree[v] && (best[v] < val || (best[v] == val && v < idx))) {
                val = best[v];
                idx = v;
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, val, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ov < val || (ov == val && oi < idx)) {
                val = ov;
                idx = oi;
            }
        }
        if (threadIdx.x % 32 == 0) {
            s_val[threadIdx.x / 32] = val;
            s_idx[threadIdx.x / 32] = idx;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float bv = inf;
            int bi = 0x7fffffff;
            for (int w = 0; w < kThreads / 32; ++w)
                if (s_val[w] < bv || (s_val[w] == bv && s_idx[w] < bi)) {
                    bv = s_val[w];
                    bi = s_idx[w];
                }
            s_pick = bi;
            s_pick_val = bv;
        }
        __syncthreads();
        const int u = s_pick;
        const float uv = s_pick_val;
        if (isinf(uv) || u == 0x7fffffff) {
            if (threadIdx.x == 0) result[0] = inf;
            return;
        }
        total += uv;
        if (threadIdx.x == 0) in_tree[u] = 1;
        for (int v = threadIdx.x; v < n; v += kThreads) {
            const float w = adj[static_cast<size_t>(u) * n + v];
            if (w != 0.0f && w < best[v]) best[v] = w;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) result[0] = static_cast<float>(total);
}

// A, min_weight are device pointers
extern "C" void solution(const float* A, float* min_weight, size_t n) {
    if (n <= 1) {
        cudaMemset(min_weight, 0, sizeof(float));
        return;
    }
    void* buf = nullptr;
    cudaMalloc(&buf, n * (sizeof(float) + 1));
    float* best = static_cast<float*>(buf);
    unsigned char* in_tree = reinterpret_cast<unsigned char*>(best + n);
    prim<<<1, kThreads>>>(A, best, in_tree, min_weight, static_cast<int>(n));
    cudaDeviceSynchronize();
    cudaFree(buf);
}

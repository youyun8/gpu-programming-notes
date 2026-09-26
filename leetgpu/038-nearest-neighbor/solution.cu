// Nearest Neighbor (LeetGPU)
// https://leetgpu.com/challenges/nearest-neighbor
//
// Brute force O(N^2) with shared-memory tiling: every thread owns one query
// point; the block streams all points through shared memory in tiles of 256
// (structure of arrays, conflict-free broadcast reads).
// The check is exact, so distances are evaluated exactly like the reference:
// d = (dx*dx + dy*dy) + dz*dz with separately rounded ops (no FMA contraction),
// and ties keep the lower index (torch.argmin semantics).
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kBlockSize = 256;

__global__ void nearestNeighbor(const float* points, int* indices, int n) {
    __shared__ float sx[kBlockSize];
    __shared__ float sy[kBlockSize];
    __shared__ float sz[kBlockSize];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    if (i < n) {
        px = points[3 * static_cast<size_t>(i)];
        py = points[3 * static_cast<size_t>(i) + 1];
        pz = points[3 * static_cast<size_t>(i) + 2];
    }
    float best = FLT_MAX;
    int best_j = -1;
    for (int t0 = 0; t0 < n; t0 += kBlockSize) {
        const int j = t0 + threadIdx.x;
        if (j < n) {
            sx[threadIdx.x] = points[3 * static_cast<size_t>(j)];
            sy[threadIdx.x] = points[3 * static_cast<size_t>(j) + 1];
            sz[threadIdx.x] = points[3 * static_cast<size_t>(j) + 2];
        }
        __syncthreads();
        const int tile = min(kBlockSize, n - t0);
        for (int t = 0; t < tile; ++t) {
            const float dx = __fsub_rn(px, sx[t]);
            const float dy = __fsub_rn(py, sy[t]);
            const float dz = __fsub_rn(pz, sz[t]);
            const float d = __fadd_rn(__fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy)), __fmul_rn(dz, dz));
            const int j2 = t0 + t;
            if (j2 != i && (d < best || best_j < 0)) {
                best = d;
                best_j = j2;
            }
        }
        __syncthreads();
    }
    if (i < n) indices[i] = best_j < 0 ? 0 : best_j;
}

// points, indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {
    nearestNeighbor<<<(N + kBlockSize - 1) / kBlockSize, kBlockSize>>>(points, indices, N);
    cudaDeviceSynchronize();
}

// All-Pairs Shortest Path (Tensara)
// https://tensara.org/problems/all-pairs-shortest-path
//
// Input: weighted adjacency matrix where 0 means "no edge" (off the diagonal);
// unreachable pairs are reported as -1. A prep kernel maps 0 -> +inf and zeroes
// the diagonal, a fix-up kernel maps +inf -> -1, and in between runs
//
// Blocked Floyd-Warshall with 32 x 32 tiles. For every block round b:
//   phase 1: the diagonal tile (b, b) runs the 32 k-steps in shared memory;
//   phase 2: tiles in row b and column b update against the diagonal tile;
//   phase 3: every other tile (i, j) does min(d_ij, d_ib + d_bj) over the 32 k
//            of the round using the two panel tiles staged in shared memory.
// Each round touches every tile once, so DRAM traffic drops by ~32x compared
// with one kernel per k. Out-of-range entries are padded with +inf.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kT = 32;

// Out-of-range cells read as +inf, so partial edge tiles need no special cases.
__device__ __forceinline__ float load(const float* d, int n, int r, int c) {
    return (r < n && c < n) ? d[static_cast<size_t>(r) * n + c] : __int_as_float(0x7f800000);
}

__global__ void phase1(float* d, int n, int b) {
    // Phase 1: the diagonal tile (b, b) is closed under its own 32 intermediates, in shared memory.
    __shared__ float t[kT][kT];
    const int r = b * kT + threadIdx.y, c = b * kT + threadIdx.x;
    t[threadIdx.y][threadIdx.x] = load(d, n, r, c);
    __syncthreads();
    // 32 dependent relaxation steps; read the candidate, sync, update, sync
    // (the tile both provides and receives values within a step).
    for (int k = 0; k < kT; ++k) {
        const float cand = t[threadIdx.y][k] + t[k][threadIdx.x];
        __syncthreads();
        if (cand < t[threadIdx.y][threadIdx.x]) t[threadIdx.y][threadIdx.x] = cand;
        __syncthreads();
    }
    // Write the finished diagonal tile back.
    if (r < n && c < n) d[static_cast<size_t>(r) * n + c] = t[threadIdx.y][threadIdx.x];
}

// blockIdx.y == 0: row panel tile (b, j); == 1: column panel tile (i, b).
__global__ void phase2(float* d, int n, int b) {
    // Phase 2: the other tiles of row b (blockIdx.y == 0) and column b (blockIdx.y == 1),
    // each relaxed through the finished diagonal tile.
    const int idx = blockIdx.x;
    if (idx == b) return;
    __shared__ float diag[kT][kT];
    __shared__ float t[kT][kT];
    const bool row_panel = blockIdx.y == 0;
    const int tr = row_panel ? b : idx;
    const int tc = row_panel ? idx : b;
    // Stage the diagonal tile and this panel tile.
    diag[threadIdx.y][threadIdx.x] = load(d, n, b * kT + threadIdx.y, b * kT + threadIdx.x);
    const int r = tr * kT + threadIdx.y, c = tc * kT + threadIdx.x;
    t[threadIdx.y][threadIdx.x] = load(d, n, r, c);
    __syncthreads();
    // Row panel: d(b, j) via diag(b, k) + d(k, j); column panel: d(i, b) via d(i, k) + diag(k, b).
    for (int k = 0; k < kT; ++k) {
        const float cand = row_panel ? diag[threadIdx.y][k] + t[k][threadIdx.x] : t[threadIdx.y][k] + diag[k][threadIdx.x];
        __syncthreads();
        if (cand < t[threadIdx.y][threadIdx.x]) t[threadIdx.y][threadIdx.x] = cand;
        __syncthreads();
    }
    // Write the panel tile back.
    if (r < n && c < n) d[static_cast<size_t>(r) * n + c] = t[threadIdx.y][threadIdx.x];
}

__global__ void phase3(float* d, int n, int b) {
    // Phase 3: every remaining tile (i, j) is a (min, +) product of its column-panel tile
    // (i, b) and row-panel tile (b, j); no dependencies inside the tile, so no inner barriers.
    const int ti = blockIdx.y, tj = blockIdx.x;
    if (ti == b || tj == b) return;
    __shared__ float col[kT][kT];  // tile (ti, b)
    __shared__ float row[kT][kT];  // tile (b, tj)
    const int r = ti * kT + threadIdx.y, c = tj * kT + threadIdx.x;
    col[threadIdx.y][threadIdx.x] = load(d, n, r, b * kT + threadIdx.x);
    row[threadIdx.y][threadIdx.x] = load(d, n, b * kT + threadIdx.y, c);
    __syncthreads();
    float best = load(d, n, r, c);
    for (int k = 0; k < kT; ++k) best = fminf(best, col[threadIdx.y][k] + row[k][threadIdx.x]);
    if (r < n && c < n) d[static_cast<size_t>(r) * n + c] = best;
}

__global__ void prepare(const float* adj, float* d, int n) {
    // Initialisation: 0 on the diagonal, +inf where there is no edge (weight 0), else the weight.
    const size_t total = static_cast<size_t>(n) * n;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const bool diag = i / n == i % n;
        d[i] = diag ? 0.0f : (adj[i] == 0.0f ? __int_as_float(0x7f800000) : adj[i]);
    }
}

__global__ void unreachableToMinusOne(float* d, int n) {
    // Output encoding: unreachable pairs become -1.
    const size_t total = static_cast<size_t>(n) * n;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        if (isinf(d[i])) d[i] = -1.0f;
}

// adj_matrix, output are device pointers
extern "C" void solution(const float* adj_matrix, float* output, size_t n) {
    const int N = static_cast<int>(n);
    // Blocked Floyd-Warshall in place in `output`: 3 launches per 32-wide round of intermediates.
    prepare<<<1024, 256>>>(adj_matrix, output, N);
    const int tiles = (N + kT - 1) / kT;
    const dim3 block(kT, kT);
    for (int b = 0; b < tiles; ++b) {
        phase1<<<1, block>>>(output, N, b);
        phase2<<<dim3(tiles, 2), block>>>(output, N, b);
        phase3<<<dim3(tiles, tiles), block>>>(output, N, b);
    }
    unreachableToMinusOne<<<1024, 256>>>(output, N);
}

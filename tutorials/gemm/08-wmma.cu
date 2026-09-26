// GEMM technique 8a: tensor cores through the WMMA API (sm_70+).
//
// C = A B with A, B in FP16 and FP32 accumulation/output. A 128 x 128 block tile,
// K slices of 32, 8 warps in a 2 x 4 layout; every warp owns a 64 x 32 warp tile =
// 4 x 2 WMMA fragments of 16 x 16 (the warp tiling of 04-warp-tiling.cu, with a
// 16 x 16 x 16 matrix instruction instead of a lane's 4 x 4 outer product).
// Slices are double-buffered with cp.async (03-cp-async.cu).
//
// WMMA fragments are opaque: which lane holds which element is not specified, so
// data goes through shared memory on the way in (load_matrix_sync) and on the way
// out (store_matrix_sync into a per-warp staging tile, then bounds-checked stores).
// 09-mma-sync.cu drops to the PTX instruction whose layout *is* specified.
//
// Requires K % 8 == 0 and N % 8 == 0 (16-byte rows for the copies); M is arbitrary.
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 08-wmma.cu -o wmma
#include <cuda_pipeline.h>
#include <mma.h>

#include "harness.cuh"

using namespace nvcuda;

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 32;
constexpr int kThreads = 256;
constexpr int kWarpsN = 4;          // 2 x 4 warps
constexpr int kWarpTileM = 64;
constexpr int kWarpTileN = 32;
constexpr int kFragsM = kWarpTileM / 16;  // 4
constexpr int kFragsN = kWarpTileN / 16;  // 2
constexpr int kStrideA = kBlockK + 8;     // 40 halves = 80 bytes: a multiple of 16 bytes (WMMA ldm rule)
constexpr int kStrideB = kBlockN + 8;     // 136 halves = 272 bytes; the +8 staggers banks
constexpr int kStrideC = 16 + 4;          // staging tile, floats

// One K slice: A is 128 x 32 halves (4 chunks of 16 bytes per row), B is 32 x 128
// (16 chunks per row); 512 chunks each, 2 of each per thread. Out-of-range chunks
// are zero-filled by the copy.
__device__ __forceinline__ void issueSlice(half (*a_s)[kStrideA], half (*b_s)[kStrideB], const half* a,
                                           const half* b, int m, int n, int k, int row0, int col0, int k0) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int chunk = threadIdx.x + i * kThreads;
        const int ar = chunk / 4, ac = (chunk % 4) * 8;
        const bool a_in = row0 + ar < m && k0 + ac < k;
        const half* a_src = a_in ? a + static_cast<size_t>(row0 + ar) * k + k0 + ac : a;
        __pipeline_memcpy_async(&a_s[ar][ac], a_src, 16, a_in ? 0 : 16);

        const int br = chunk / 16, bc = (chunk % 16) * 8;
        const bool b_in = k0 + br < k && col0 + bc < n;
        const half* b_src = b_in ? b + static_cast<size_t>(k0 + br) * n + col0 + bc : b;
        __pipeline_memcpy_async(&b_s[br][bc], b_src, 16, b_in ? 0 : 16);
    }
}

__global__ void __launch_bounds__(kThreads) hgemmWmma(const half* __restrict__ a, const half* __restrict__ b,
                                                      float* __restrict__ c, int m, int n, int k) {
    // WMMA needs 32-byte-aligned fragment pointers: every row offset used below is a
    // multiple of 32 bytes (16 rows x 80 or 272 bytes, and 16-column steps of 32 bytes).
    __shared__ __align__(128) half a_s[2][kBlockM][kStrideA];
    __shared__ __align__(128) half b_s[2][kBlockK][kStrideB];
    __shared__ __align__(128) float c_stage[kThreads / 32][16][kStrideC];

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int warp_m = warp / kWarpsN, warp_n = warp % kWarpsN;
    const int row0 = blockIdx.y * kBlockM;
    const int col0 = blockIdx.x * kBlockN;
    const int num_slices = (k + kBlockK - 1) / kBlockK;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kFragsM][kFragsN];
#pragma unroll
    for (int i = 0; i < kFragsM; ++i)
#pragma unroll
        for (int j = 0; j < kFragsN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    issueSlice(a_s[0], b_s[0], a, b, m, n, k, row0, col0, 0);
    __pipeline_commit();

    for (int s = 0; s < num_slices; ++s) {
        const int buf = s % 2;
        // Start the next slice, then wait for everything but it: slice s has landed.
        if (s + 1 < num_slices) issueSlice(a_s[buf ^ 1], b_s[buf ^ 1], a, b, m, n, k, row0, col0, (s + 1) * kBlockK);
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kBlockK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[kFragsM];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[kFragsN];
#pragma unroll
            for (int i = 0; i < kFragsM; ++i)
                wmma::load_matrix_sync(a_frag[i], &a_s[buf][warp_m * kWarpTileM + 16 * i][kk], kStrideA);
#pragma unroll
            for (int j = 0; j < kFragsN; ++j)
                wmma::load_matrix_sync(b_frag[j], &b_s[buf][kk][warp_n * kWarpTileN + 16 * j], kStrideB);
            // 8 tensor-core MMAs per k16 step; each A fragment is reused twice, each B fragment 4 times.
#pragma unroll
            for (int i = 0; i < kFragsM; ++i)
#pragma unroll
                for (int j = 0; j < kFragsN; ++j) wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        // Everyone must be done with buffer `buf` before iteration s + 1 refills it.
        __syncthreads();
    }

    // Epilogue: fragment -> per-warp staging tile -> C with bounds checks.
#pragma unroll
    for (int i = 0; i < kFragsM; ++i)
#pragma unroll
        for (int j = 0; j < kFragsN; ++j) {
            wmma::store_matrix_sync(&c_stage[warp][0][0], acc[i][j], kStrideC, wmma::mem_row_major);
            __syncwarp();
            const int tile_row = row0 + warp_m * kWarpTileM + 16 * i;
            const int tile_col = col0 + warp_n * kWarpTileN + 16 * j;
#pragma unroll
            for (int e = lane; e < 256; e += 32) {  // lanes 0-15 write row r, lanes 16-31 row r+1
                const int r = e / 16, cc = e % 16;
                if (tile_row + r < m && tile_col + cc < n)
                    c[static_cast<size_t>(tile_row + r) * n + tile_col + cc] = c_stage[warp][r][cc];
            }
            __syncwarp();  // the staging tile is reused by the next fragment
        }
}

void launchWmma(const half* a, const half* b, float* c, int m, int n, int k) {
    if (k % 8 != 0 || n % 8 != 0) {
        std::fprintf(stderr, "08-wmma needs K %% 8 == 0 and N %% 8 == 0 (got N=%d K=%d)\n", n, k);
        std::exit(1);
    }
    const dim3 grid(gemm::ceilDiv(n, kBlockN), gemm::ceilDiv(m, kBlockM));
    hgemmWmma<<<grid, kThreads>>>(a, b, c, m, n, k);
}

// Test shapes with K and N multiples of 8.
inline std::vector<gemm::Shape> halfTestShapes() {
    return {{128, 128, 64}, {256, 128, 40}, {67, 48, 32}, {1, 136, 24}, {130, 264, 96}, {200, 8, 136}};
}

int main(int argc, char** argv) {
    return gemm::runMain<half>("08-wmma", launchWmma, argc, argv, halfTestShapes());
}

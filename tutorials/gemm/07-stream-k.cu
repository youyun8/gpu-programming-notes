// GEMM technique 7: Stream-K (Osama et al., 2023), with a deterministic fix-up.
//
// Split the whole GEMM into L = tiles x ceil(K / kBlockK) MAC-loop iterations, laid
// out tile after tile, and give each of G persistent blocks (G = what fits on the GPU
// at once) a contiguous range of L / G iterations. A block's range can start in the
// middle of one tile and end in the middle of another:
//
//   iterations: |---- tile 0 ----|---- tile 1 ----|---- tile 2 ----|...
//   blocks:     |-- block 0 --|-- block 1 --|-- block 2 --|-- block 3 --|...
//
// For every tile it touches, a block accumulates a partial result over its part of K:
//
//   whole tile in range      -> store the tile to C
//   range ends inside tile   -> "contributor": store the partial to workspace[block],
//                               then raise flags[block]
//   range ends at tile's end, but the tile began in an earlier block's range
//                            -> "owner": wait for the contributors' flags, add their
//                               partials in a fixed order, store the tile
//
// Only the last tile of a range can be a contributor tile, so one workspace slot per
// block suffices. An owner only waits for blocks with smaller indices, and all G
// blocks are resident at once, so the wait cannot deadlock.
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 07-stream-k.cu -o stream_k
#include "harness.cuh"

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 8;
constexpr int kThreads = 256;
constexpr int kPadA = 4;  // keeps the transposed stores of A conflict-free and rows 16-byte aligned

// Loads 4 consecutive floats of row `row` starting at column `col` (col % 4 == 0),
// with zeros outside the matrix. kVec: the row length is a multiple of 4, so the
// 4 floats are 16-byte aligned and either all inside or all outside.
template <bool kVec>
__device__ __forceinline__ float4 load4(const float* p, int rows, int cols, int row, int col) {
    if (row >= rows) return make_float4(0.f, 0.f, 0.f, 0.f);
    const float* src = p + static_cast<size_t>(row) * cols + col;
    if (kVec) return col < cols ? *reinterpret_cast<const float4*>(src) : make_float4(0.f, 0.f, 0.f, 0.f);
    return make_float4(col < cols ? src[0] : 0.f, col + 1 < cols ? src[1] : 0.f, col + 2 < cols ? src[2] : 0.f,
                       col + 3 < cols ? src[3] : 0.f);
}

template <bool kVec>
__device__ __forceinline__ void store4(float* p, int rows, int cols, int row, int col, float4 v) {
    if (row >= rows) return;
    float* dst = p + static_cast<size_t>(row) * cols + col;
    if (kVec) {
        if (col < cols) *reinterpret_cast<float4*>(dst) = v;
        return;
    }
    if (col < cols) dst[0] = v.x;
    if (col + 1 < cols) dst[1] = v.y;
    if (col + 2 < cols) dst[2] = v.z;
    if (col + 3 < cols) dst[3] = v.w;
}

constexpr int kTileElems = kBlockM * kBlockN;

// First iteration of block g's range: floor(g L / G), in 64 bits.
__host__ __device__ inline long long rangeStart(int g, long long total, int num_blocks) {
    return static_cast<long long>(g) * total / num_blocks;
}

template <bool kVec>
__global__ void __launch_bounds__(kThreads) sgemmStreamK(const float* __restrict__ a, const float* __restrict__ b,
                                                         float* __restrict__ c, int m, int n, int k,
                                                         float* __restrict__ workspace, int* flags) {
    __shared__ __align__(16) float a_s[kBlockK][kBlockM + kPadA];
    __shared__ __align__(16) float b_s[kBlockK][kBlockN];

    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int a_row = tid / 2, a_col = (tid % 2) * 4;
    const int b_row = tid / 32, b_col = (tid % 32) * 4;

    const int tiles_n = (n + kBlockN - 1) / kBlockN;
    const int tiles = ((m + kBlockM - 1) / kBlockM) * tiles_n;
    const int iters_per_tile = (k + kBlockK - 1) / kBlockK;
    const long long total = static_cast<long long>(tiles) * iters_per_tile;
    const int g = blockIdx.x;
    const long long end = rangeStart(g + 1, total, gridDim.x);

    for (long long it = rangeStart(g, total, gridDim.x); it < end;) {
        const int tile = static_cast<int>(it / iters_per_tile);
        const long long tile_begin = static_cast<long long>(tile) * iters_per_tile;
        const long long tile_end = tile_begin + iters_per_tile;
        const long long seg_end = end < tile_end ? end : tile_end;
        const int row0 = (tile / tiles_n) * kBlockM;
        const int col0 = (tile % tiles_n) * kBlockN;

        // ---- MAC loop over this block's part of the tile (the loop of 01-vectorized.cu).
        float acc[8][8] = {};
        for (long long i = it; i < seg_end; ++i) {
            const int k0 = static_cast<int>(i - tile_begin) * kBlockK;
            const float4 av = load4<kVec>(a, m, k, row0 + a_row, k0 + a_col);
            const float4 bv = load4<kVec>(b, k, n, k0 + b_row, col0 + b_col);
            a_s[a_col + 0][a_row] = av.x;
            a_s[a_col + 1][a_row] = av.y;
            a_s[a_col + 2][a_row] = av.z;
            a_s[a_col + 3][a_row] = av.w;
            *reinterpret_cast<float4*>(&b_s[b_row][b_col]) = bv;
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < kBlockK; ++kk) {
                const float4 a_lo = *reinterpret_cast<const float4*>(&a_s[kk][4 * ty]);
                const float4 a_hi = *reinterpret_cast<const float4*>(&a_s[kk][64 + 4 * ty]);
                const float4 b_lo = *reinterpret_cast<const float4*>(&b_s[kk][4 * tx]);
                const float4 b_hi = *reinterpret_cast<const float4*>(&b_s[kk][64 + 4 * tx]);
                const float a_frag[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w, a_hi.x, a_hi.y, a_hi.z, a_hi.w};
                const float b_frag[8] = {b_lo.x, b_lo.y, b_lo.z, b_lo.w, b_hi.x, b_hi.y, b_hi.z, b_hi.w};
#pragma unroll
                for (int i2 = 0; i2 < 8; ++i2)
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc[i2][j] = fmaf(a_frag[i2], b_frag[j], acc[i2][j]);
            }
            __syncthreads();
        }

        if (seg_end < tile_end) {
            // ---- Contributor: publish the partial tile. Element (i, j) of this thread goes
            // to workspace[g][(8 i + j) * kThreads + tid]: consecutive threads, consecutive words.
            float* slot = workspace + static_cast<size_t>(g) * kTileElems;
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) slot[(8 * i + j) * kThreads + tid] = acc[i][j];
            __threadfence();  // make the partial visible device-wide before the flag
            __syncthreads();  // ... for every thread of the block
            if (tid == 0) atomicExch(&flags[g], 1);
        } else {
            if (it > tile_begin) {
                // ---- Owner of a tile that started in earlier blocks: add their partials.
                // Contributors are the blocks g-1, g-2, ... whose ranges end after tile_begin.
                for (int p = g - 1; p >= 0 && rangeStart(p + 1, total, gridDim.x) > tile_begin; --p) {
                    if (rangeStart(p, total, gridDim.x) == rangeStart(p + 1, total, gridDim.x)) continue;  // empty range
                    if (tid == 0) {
                        while (atomicAdd(&flags[p], 0) == 0) {
                        }
                        __threadfence();  // acquire: order the reads below after the flag
                    }
                    __syncthreads();
                    // The partial was written by another SM. Volatile loads (relaxed, system
                    // scope) cannot be served from a stale line in this SM's L1, which is not
                    // coherent with other SMs' writes.
                    const volatile float* slot = workspace + static_cast<size_t>(p) * kTileElems;
#pragma unroll
                    for (int i = 0; i < 8; ++i)
#pragma unroll
                        for (int j = 0; j < 8; ++j) acc[i][j] += slot[(8 * i + j) * kThreads + tid];
                }
            }
            // ---- The tile is complete: store it.
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = row0 + (i < 4 ? 4 * ty + i : 64 + 4 * ty + (i - 4));
                store4<kVec>(c, m, n, row, col0 + 4 * tx, make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]));
                store4<kVec>(c, m, n, row, col0 + 64 + 4 * tx,
                             make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]));
            }
        }
        it = seg_end;
    }
}

static int g_blocks = 0;  // 0: one full wave of resident blocks

void launchStreamK(const float* a, const float* b, float* c, int m, int n, int k) {
    const bool vec = k % 4 == 0 && n % 4 == 0;
    // G = SMs x resident blocks per SM: every block is running at the same time.
    int num_sms = 0, per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    if (vec)
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, sgemmStreamK<true>, kThreads, 0));
    else
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, sgemmStreamK<false>, kThreads, 0));
    const long long total = static_cast<long long>(gemm::ceilDiv(m, kBlockM)) * gemm::ceilDiv(n, kBlockN) *
                            gemm::ceilDiv(k, kBlockK);
    int blocks = g_blocks > 0 ? g_blocks : num_sms * per_sm;
    if (blocks > total) blocks = static_cast<int>(total);  // no empty ranges needed

    // A real library keeps these buffers around; allocating per call keeps the example short.
    float* workspace = nullptr;
    int* flags = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, static_cast<size_t>(blocks) * kTileElems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&flags, blocks * sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(flags, 0, blocks * sizeof(int)));
    if (vec)
        sgemmStreamK<true><<<blocks, kThreads>>>(a, b, c, m, n, k, workspace, flags);
    else
        sgemmStreamK<false><<<blocks, kThreads>>>(a, b, c, m, n, k, workspace, flags);
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(flags));
}

int main(int argc, char** argv) {
    std::vector<char*> args{argv[0]};
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--blocks=", 9) == 0)
            g_blocks = std::atoi(argv[i] + 9);
        else
            args.push_back(argv[i]);
    }
    std::vector<gemm::Shape> shapes = gemm::defaultTestShapes();
    shapes.push_back({64, 64, 1000});
    shapes.push_back({300, 260, 200});
    return gemm::runMain<float>("07-stream-k", launchStreamK, static_cast<int>(args.size()), args.data(), shapes);
}

// GEMM technique 4: warp tiling (on top of float4 accesses and double buffering).
//
// The 128 x 128 block tile is split into 2 x 4 warp tiles of 64 x 32, and each warp
// tile into 2 x 2 sub-tiles of 32 x 16 that its 32 lanes cover as an 8 x 4 grid of
// 4 x 4 patches. A lane therefore owns 2 x 2 patches = 8 x 8 outputs, like in
// 01-vectorized.cu, but at positions chosen by the warp:
//
//   block tile 128 x 128  ->  warp tile 64 x 32  ->  sub-tile 32 x 16  ->  lane 4 x 4
//
// Why: the shared-memory traffic of a warp is set by the *warp's* footprint. Per k
// step a warp reads 64 + 32 = 96 distinct floats here (the 16 x 2 thread layout of
// 01 reads 16 + 128 = 144), and each lane's fragments are 2 float4 of A and 2 float4
// of B, broadcast among lanes that share a row or column. The same hierarchy maps
// one-to-one onto tensor-core code, where a warp is the unit that issues MMAs.
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 04-warp-tiling.cu -o warp_tiling
#include "harness.cuh"

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 8;
constexpr int kThreads = 256;
constexpr int kWarpsM = 2;              // warps along M
constexpr int kWarpsN = 4;              // warps along N
constexpr int kWarpTileM = kBlockM / kWarpsM;  // 64
constexpr int kWarpTileN = kBlockN / kWarpsN;  // 32
constexpr int kLanesM = 8;              // lanes of a warp along M
constexpr int kLanesN = 4;              // lanes of a warp along N
constexpr int kThreadM = 4;             // one sub-tile: 4 x 4 outputs per lane
constexpr int kThreadN = 4;
constexpr int kIterM = kWarpTileM / (kLanesM * kThreadM);  // 2 sub-tiles along M
constexpr int kIterN = kWarpTileN / (kLanesN * kThreadN);  // 2 sub-tiles along N
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

template <bool kVec>
__global__ void __launch_bounds__(kThreads) sgemmWarpTiled(const float* __restrict__ a, const float* __restrict__ b,
                                                           float* __restrict__ c, int m, int n, int k) {
    __shared__ __align__(16) float a_s[2][kBlockK][kBlockM + kPadA];  // transposed
    __shared__ __align__(16) float b_s[2][kBlockK][kBlockN];

    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;
    const int warp_m = warp / kWarpsN, warp_n = warp % kWarpsN;  // warp position in the block tile
    const int lane_m = lane / kLanesN, lane_n = lane % kLanesN;  // lane position in a sub-tile
    const int row0 = blockIdx.y * kBlockM;
    const int col0 = blockIdx.x * kBlockN;
    // Global -> shared mapping: identical to 01/02 (one float4 of A and of B per thread).
    const int a_row = tid / 2, a_col = (tid % 2) * 4;
    const int b_row = tid / 32, b_col = (tid % 32) * 4;

    // First row / column of the lane's patch in sub-tile (0, 0), relative to the block tile.
    const int m_base = warp_m * kWarpTileM + lane_m * kThreadM;
    const int n_base = warp_n * kWarpTileN + lane_n * kThreadN;

    auto storeSlice = [&](int buf, float4 av, float4 bv) {
        a_s[buf][a_col + 0][a_row] = av.x;
        a_s[buf][a_col + 1][a_row] = av.y;
        a_s[buf][a_col + 2][a_row] = av.z;
        a_s[buf][a_col + 3][a_row] = av.w;
        *reinterpret_cast<float4*>(&b_s[buf][b_row][b_col]) = bv;
    };

    float acc[kIterM * kThreadM][kIterN * kThreadN] = {};
    const int num_slices = (k + kBlockK - 1) / kBlockK;
    storeSlice(0, load4<kVec>(a, m, k, row0 + a_row, a_col), load4<kVec>(b, k, n, b_row, col0 + b_col));
    __syncthreads();

    for (int s = 0; s < num_slices; ++s) {
        const int buf = s % 2;
        const bool has_next = s + 1 < num_slices;
        float4 a_next = make_float4(0.f, 0.f, 0.f, 0.f), b_next = a_next;
        if (has_next) {
            const int k1 = (s + 1) * kBlockK;
            a_next = load4<kVec>(a, m, k, row0 + a_row, k1 + a_col);
            b_next = load4<kVec>(b, k, n, k1 + b_row, col0 + b_col);
        }
#pragma unroll
        for (int kk = 0; kk < kBlockK; ++kk) {
            // One float4 per sub-tile: sub-tile im starts kLanesM * kThreadM = 32 rows further.
            float a_frag[kIterM * kThreadM], b_frag[kIterN * kThreadN];
#pragma unroll
            for (int im = 0; im < kIterM; ++im) {
                const float4 v = *reinterpret_cast<const float4*>(&a_s[buf][kk][m_base + im * kLanesM * kThreadM]);
                a_frag[4 * im + 0] = v.x;
                a_frag[4 * im + 1] = v.y;
                a_frag[4 * im + 2] = v.z;
                a_frag[4 * im + 3] = v.w;
            }
#pragma unroll
            for (int in = 0; in < kIterN; ++in) {
                const float4 v = *reinterpret_cast<const float4*>(&b_s[buf][kk][n_base + in * kLanesN * kThreadN]);
                b_frag[4 * in + 0] = v.x;
                b_frag[4 * in + 1] = v.y;
                b_frag[4 * in + 2] = v.z;
                b_frag[4 * in + 3] = v.w;
            }
#pragma unroll
            for (int i = 0; i < kIterM * kThreadM; ++i)
#pragma unroll
                for (int j = 0; j < kIterN * kThreadN; ++j) acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
        }
        if (has_next) storeSlice(buf ^ 1, a_next, b_next);
        __syncthreads();
    }

    // Epilogue: acc[4 im + i][4 in + j] is element (m_base + 32 im + i, n_base + 16 in + j).
#pragma unroll
    for (int im = 0; im < kIterM; ++im)
#pragma unroll
        for (int i = 0; i < kThreadM; ++i) {
            const int row = row0 + m_base + im * kLanesM * kThreadM + i;
#pragma unroll
            for (int in = 0; in < kIterN; ++in) {
                const float* v = &acc[4 * im + i][4 * in];
                store4<kVec>(c, m, n, row, col0 + n_base + in * kLanesN * kThreadN,
                             make_float4(v[0], v[1], v[2], v[3]));
            }
        }
}

void launchWarpTiled(const float* a, const float* b, float* c, int m, int n, int k) {
    const dim3 grid(gemm::ceilDiv(n, kBlockN), gemm::ceilDiv(m, kBlockM));
    if (k % 4 == 0 && n % 4 == 0)
        sgemmWarpTiled<true><<<grid, kThreads>>>(a, b, c, m, n, k);
    else
        sgemmWarpTiled<false><<<grid, kThreads>>>(a, b, c, m, n, k);
}

int main(int argc, char** argv) {
    return gemm::runMain<float>("04-warp-tiling", launchWarpTiled, argc, argv);
}

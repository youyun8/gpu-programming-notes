// GEMM technique 8b: tensor cores through PTX: ldmatrix + mma.sync (sm_80+).
//
// C = A B with A, B in FP16, FP32 accumulation/output, and everything from the
// previous pages combined:
//
//   block tile 128 x 128 x 32, 8 warps (2 x 4), warp tile 64 x 32   (warp tiling)
//   3-stage cp.async pipeline, one barrier per K slice               (async copies)
//   XOR-swizzled shared-memory layout: no padding, no bank conflicts (swizzling)
//   ldmatrix: one instruction loads a 16 x 16 A fragment (x4) or two 16 x 8 B
//             fragments (x4.trans) straight into the mma.sync register layout
//   mma.sync.m16n8k16: 4 x 4 = 16 per k16 step per warp, FP32 accumulators in registers
//
// Unlike WMMA, the PTX fragment layouts are documented (PTX ISA, "Matrix fragments
// for mma.m16n8k16"), so the epilogue writes straight from registers to C. With
// g = lane / 4 and t = lane % 4, a warp's accumulator tile d[0..3] is
//
//   d0, d1 = C[g][2t], C[g][2t+1]        d2, d3 = C[g+8][2t], C[g+8][2t+1]
//
// The #ifdef __CUEMU__ branches let the CPU emulator (tools/cuemu) check this file:
// it implements the same two instructions with the documented layouts.
//
// Requires K % 8 == 0 and N % 8 == 0 (16-byte rows for the copies); M is arbitrary.
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 09-mma-sync.cu -o mma_sync
#include <cuda_pipeline.h>

#include "harness.cuh"

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 32;
constexpr int kThreads = 256;
constexpr int kStages = 3;          // 3 x (8 KiB of A + 8 KiB of B) = 48 KiB of shared memory
constexpr int kWarpsN = 4;          // 2 x 4 warps
constexpr int kWarpTileM = 64;
constexpr int kWarpTileN = 32;
constexpr int kTilesM = kWarpTileM / 16;  // 4 m16 tiles per warp
constexpr int kTilesN = kWarpTileN / 8;   // 4 n8 tiles per warp

// ---- Swizzled shared-memory layouts --------------------------------------------------
// Shared memory has 32 banks of 4 bytes; a 16-byte chunk covers 4 banks, so 8 chunks
// that start in 8 different 4-bank groups are conflict-free. ldmatrix reads 8 rows of
// one 16-byte chunk column per 8 x 8 matrix, so the chunk column is XORed with bits of
// the row index to spread those 8 rows over all 8 groups.
//
// A slice: 128 rows x 32 halves = 4 chunks (64 bytes) per row. Two rows share one
// 128-byte bank line, so row r starts in bank group 4 (r % 2); XOR with (r / 2) % 4
// supplies the other two bits.
__device__ __forceinline__ int offsetA(int row, int col) {
    const int chunk = (col / 8) ^ ((row >> 1) & 3);
    return row * kBlockK + chunk * 8 + col % 8;
}
// B slice: 32 rows x 128 halves = 16 chunks (256 bytes) per row, so every row starts in
// bank group 0; XOR with r % 8 spreads 8 consecutive rows over the 8 groups.
__device__ __forceinline__ int offsetB(int row, int col) {
    const int chunk = (col / 8) ^ (row & 7);
    return row * kBlockN + chunk * 8 + col % 8;
}

// ---- The two warp-wide instructions --------------------------------------------------
// ldmatrix.x4: lanes 8q .. 8q+7 give the addresses of the 8 rows (16 bytes each) of
// matrix q; register q of lane l receives row l/4, halves 2(l%4) and 2(l%4)+1 of
// matrix q (.trans: of its transpose).
template <bool kTrans>
__device__ __forceinline__ void ldmatrixX4(uint32_t (&r)[4], const half* row_ptr) {
#ifdef __CUEMU__
    cuemuLdmatrix(r, 4, kTrans, row_ptr);
#else
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(row_ptr));
    if (kTrans)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                     : "r"(addr));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                     : "r"(addr));
#endif
}

// d += A (16 x 16, row-major fragment) x B (16 x 8, "col" fragment), FP32 accumulators.
__device__ __forceinline__ void mmaM16N8K16(float (&d)[4], const uint32_t (&a)[4], const uint32_t* b) {
#ifdef __CUEMU__
    cuemuMmaM16N8K16(d, a, b, d);
#else
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// ---- Global -> shared: one K slice, 2 chunks of A and 2 of B per thread -----------------
__device__ __forceinline__ void issueSlice(half* a_s, half* b_s, const half* a, const half* b, int m, int n, int k,
                                           int row0, int col0, int k0) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int chunk = threadIdx.x + i * kThreads;
        const int ar = chunk / 4, ac = (chunk % 4) * 8;
        const bool a_in = row0 + ar < m && k0 + ac < k;
        const half* a_src = a_in ? a + static_cast<size_t>(row0 + ar) * k + k0 + ac : a;
        __pipeline_memcpy_async(a_s + offsetA(ar, ac), a_src, 16, a_in ? 0 : 16);

        const int br = chunk / 16, bc = (chunk % 16) * 8;
        const bool b_in = k0 + br < k && col0 + bc < n;
        const half* b_src = b_in ? b + static_cast<size_t>(k0 + br) * n + col0 + bc : b;
        __pipeline_memcpy_async(b_s + offsetB(br, bc), b_src, 16, b_in ? 0 : 16);
    }
}

__global__ void __launch_bounds__(kThreads) hgemmMmaSync(const half* __restrict__ a, const half* __restrict__ b,
                                                         float* __restrict__ c, int m, int n, int k) {
    __shared__ __align__(128) half a_s[kStages][kBlockM * kBlockK];
    __shared__ __align__(128) half b_s[kStages][kBlockK * kBlockN];

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int warp_row = (warp / kWarpsN) * kWarpTileM;  // warp tile origin inside the block tile
    const int warp_col = (warp % kWarpsN) * kWarpTileN;
    const int row0 = blockIdx.y * kBlockM;
    const int col0 = blockIdx.x * kBlockN;
    const int num_slices = (k + kBlockK - 1) / kBlockK;

    float acc[kTilesM][kTilesN][4] = {};

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < num_slices) issueSlice(a_s[s], b_s[s], a, b, m, n, k, row0, col0, s * kBlockK);
        __pipeline_commit();
    }

    for (int s = 0; s < num_slices; ++s) {
        __pipeline_wait_prior(kStages - 2);  // slice s has landed (this thread's copies) ...
        __syncthreads();                     // ... for all threads; stage of slice s-1 is free
        const int next = s + kStages - 1;
        if (next < num_slices) {
            const int st = next % kStages;
            issueSlice(a_s[st], b_s[st], a, b, m, n, k, row0, col0, next * kBlockK);
        }
        __pipeline_commit();

        const half* as = a_s[s % kStages];
        const half* bs = b_s[s % kStages];
#pragma unroll
        for (int kk = 0; kk < kBlockK; kk += 16) {
            // A: one ldmatrix.x4 per m16 tile. Matrices 0..3 = (rows 0-7, k 0-7), (rows 8-15, k 0-7),
            // (rows 0-7, k 8-15), (rows 8-15, k 8-15): exactly registers a0..a3 of mma.m16n8k16.
            uint32_t a_frag[kTilesM][4];
#pragma unroll
            for (int i = 0; i < kTilesM; ++i) {
                const int row = warp_row + 16 * i + lane % 8 + 8 * ((lane / 8) % 2);
                const int col = kk + 8 * (lane / 16);
                ldmatrixX4<false>(a_frag[i], as + offsetA(row, col));
            }
            // B: one ldmatrix.x4.trans per pair of n8 tiles. B is stored k-major (row = k), so
            // the transpose gives each lane B[2t..2t+1][g]: registers b0, b1 of two n8 tiles.
            uint32_t b_frag[kTilesN][2];
#pragma unroll
            for (int p = 0; p < kTilesN / 2; ++p) {
                const int q = lane / 8;
                const int row = kk + lane % 8 + 8 * (q % 2);
                const int col = warp_col + 16 * p + 8 * (q / 2);
                uint32_t r[4];
                ldmatrixX4<true>(r, bs + offsetB(row, col));
                b_frag[2 * p][0] = r[0];
                b_frag[2 * p][1] = r[1];
                b_frag[2 * p + 1][0] = r[2];
                b_frag[2 * p + 1][1] = r[3];
            }
#pragma unroll
            for (int i = 0; i < kTilesM; ++i)
#pragma unroll
                for (int j = 0; j < kTilesN; ++j) mmaM16N8K16(acc[i][j], a_frag[i], b_frag[j]);
        }
    }
    __pipeline_wait_prior(0);

    // Epilogue straight from the documented accumulator layout: two float2 per n8 tile.
    const int g = lane / 4, t = lane % 4;
#pragma unroll
    for (int i = 0; i < kTilesM; ++i)
#pragma unroll
        for (int j = 0; j < kTilesN; ++j) {
            const int col = col0 + warp_col + 8 * j + 2 * t;  // even, and N % 8 == 0: col < n => col + 1 < n
            if (col >= n) continue;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = row0 + warp_row + 16 * i + g + 8 * h;
                if (row < m)
                    *reinterpret_cast<float2*>(&c[static_cast<size_t>(row) * n + col]) =
                        make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
            }
        }
}

void launchMmaSync(const half* a, const half* b, float* c, int m, int n, int k) {
    if (k % 8 != 0 || n % 8 != 0) {
        std::fprintf(stderr, "09-mma-sync needs K %% 8 == 0 and N %% 8 == 0 (got N=%d K=%d)\n", n, k);
        std::exit(1);
    }
    const dim3 grid(gemm::ceilDiv(n, kBlockN), gemm::ceilDiv(m, kBlockM));
    hgemmMmaSync<<<grid, kThreads>>>(a, b, c, m, n, k);
}

int main(int argc, char** argv) {
    const std::vector<gemm::Shape> shapes = {{128, 128, 64}, {256, 128, 40}, {67, 48, 32},
                                             {1, 136, 24},   {130, 264, 96}, {200, 8, 136}};
    return gemm::runMain<half>("09-mma-sync", launchMmaSync, argc, argv, shapes);
}

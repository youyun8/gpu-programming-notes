// GEMM technique 6: split-K.
//
// When M x N has too few 128 x 128 tiles to occupy every SM (M = N = 512 is 16 tiles
// on a 132-SM GPU), split the K loop too: block (x, y, z) computes the partial
// product of its tile over the z-th range of K,
//
//   C = sum_z A[:, K_z] B[K_z, :],     K_z = [z * chunk, min(K, (z + 1) * chunk))
//
// and the S partial tiles are combined in one of two ways:
//
//   kAtomic     atomicAdd into C (zeroed first). One kernel; the FP32 summation order,
//               and so the last bits of the result, vary from run to run.
//   kWorkspace  each split writes its own M x N slice of a workspace; a second kernel
//               sums the S slices in a fixed order. Deterministic; S x M x N x 4 extra
//               bytes of traffic.
//
// The main loop is the one of 01-vectorized.cu restricted to a range of K.
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 06-split-k.cu -o split_k
//        ./split_k [--atomic] [--splits=S] [--test | M N K]
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

enum class Reduce { kAtomic, kWorkspace };

template <bool kVec, Reduce kReduce>
__global__ void __launch_bounds__(kThreads) sgemmSplitK(const float* __restrict__ a, const float* __restrict__ b,
                                                        float* __restrict__ out, int m, int n, int k,
                                                        int k_chunk) {
    __shared__ __align__(16) float a_s[kBlockK][kBlockM + kPadA];
    __shared__ __align__(16) float b_s[kBlockK][kBlockN];

    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int row0 = blockIdx.y * kBlockM;
    const int col0 = blockIdx.x * kBlockN;
    const int a_row = tid / 2, a_col = (tid % 2) * 4;
    const int b_row = tid / 32, b_col = (tid % 32) * 4;
    // This split's range of K. k_chunk is a multiple of kBlockK, so every slice but the
    // last one of the whole matrix is full, and float4 loads stay aligned.
    const int k_begin = blockIdx.z * k_chunk;
    const int k_end = min(k, k_begin + k_chunk);

    float acc[8][8] = {};
    for (int k0 = k_begin; k0 < k_end; k0 += kBlockK) {
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
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }

    // kAtomic: out is C. kWorkspace: out is the workspace, split z owns slice z.
    float* dst_base = kReduce == Reduce::kAtomic ? out : out + static_cast<size_t>(blockIdx.z) * m * n;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row0 + (i < 4 ? 4 * ty + i : 64 + 4 * ty + (i - 4));
        if (row >= m) continue;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = col0 + (j < 4 ? 4 * tx + j : 64 + 4 * tx + (j - 4));
            if (col >= n) continue;
            float* dst = dst_base + static_cast<size_t>(row) * n + col;
            if (kReduce == Reduce::kAtomic)
                atomicAdd(dst, acc[i][j]);  // RED.E.ADD.F32: resolved in L2, no return value
            else
                *dst = acc[i][j];
        }
    }
}

// C[i] = sum over z of workspace[z][i], always in the order z = 0, 1, ..., S-1.
__global__ void reduceSplits(const float* __restrict__ workspace, float* __restrict__ c, size_t count, int splits) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
        float sum = 0.0f;
        for (int z = 0; z < splits; ++z) sum += workspace[z * count + i];
        c[i] = sum;
    }
}

// Picks S so that tiles x S roughly fills `num_sms` twice, with at least 4 slices of K per split.
int chooseSplits(int m, int n, int k, int num_sms) {
    const int tiles = gemm::ceilDiv(m, kBlockM) * gemm::ceilDiv(n, kBlockN);
    const int by_fill = gemm::ceilDiv(2 * num_sms, tiles);
    const int by_depth = std::max<int>(1, k / (4 * kBlockK));
    return std::max<int>(1, std::min<int>(by_fill, by_depth));
}

static Reduce g_mode = Reduce::kWorkspace;
static int g_splits = 0;  // 0: choose automatically

void launchSplitK(const float* a, const float* b, float* c, int m, int n, int k) {
    int num_sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    const int splits_wanted = g_splits > 0 ? g_splits : chooseSplits(m, n, k, num_sms);
    // Round the chunk up to whole slices, then recompute S so that no split is empty.
    const int k_chunk = gemm::ceilDiv(gemm::ceilDiv(k, splits_wanted), kBlockK) * kBlockK;
    const int splits = gemm::ceilDiv(k, k_chunk);
    const dim3 grid(gemm::ceilDiv(n, kBlockN), gemm::ceilDiv(m, kBlockM), splits);
    const bool vec = k % 4 == 0 && n % 4 == 0;
    if (g_mode == Reduce::kAtomic) {
        CUDA_CHECK(cudaMemsetAsync(c, 0, static_cast<size_t>(m) * n * sizeof(float)));
        if (vec)
            sgemmSplitK<true, Reduce::kAtomic><<<grid, kThreads>>>(a, b, c, m, n, k, k_chunk);
        else
            sgemmSplitK<false, Reduce::kAtomic><<<grid, kThreads>>>(a, b, c, m, n, k, k_chunk);
        return;
    }
    // A real library keeps the workspace around; allocating per call keeps the example short.
    float* workspace = nullptr;
    const size_t count = static_cast<size_t>(m) * n;
    CUDA_CHECK(cudaMalloc(&workspace, count * splits * sizeof(float)));
    if (vec)
        sgemmSplitK<true, Reduce::kWorkspace><<<grid, kThreads>>>(a, b, workspace, m, n, k, k_chunk);
    else
        sgemmSplitK<false, Reduce::kWorkspace><<<grid, kThreads>>>(a, b, workspace, m, n, k, k_chunk);
    const int reduce_blocks = static_cast<int>(std::min<size_t>((count + 255) / 256, 4096));
    reduceSplits<<<reduce_blocks, 256>>>(workspace, c, count, splits);
    CUDA_CHECK(cudaFree(workspace));  // cudaFree synchronizes, so the kernels are done
}

int main(int argc, char** argv) {
    // Strip this example's own options, pass the rest to the harness.
    std::vector<char*> args{argv[0]};
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--atomic") == 0)
            g_mode = Reduce::kAtomic;
        else if (std::strncmp(argv[i], "--splits=", 9) == 0)
            g_splits = std::atoi(argv[i] + 9);
        else
            args.push_back(argv[i]);
    }
    const char* name = g_mode == Reduce::kAtomic ? "06-split-k (atomic)" : "06-split-k (workspace)";
    // Skinny-output shapes are where split-K matters; test them as well as the defaults.
    std::vector<gemm::Shape> shapes = gemm::defaultTestShapes();
    shapes.push_back({64, 64, 1000});
    shapes.push_back({33, 130, 777});
    return gemm::runMain<float>(name, launchSplitK, static_cast<int>(args.size()), args.data(), shapes);
}

// Chapter 09: kernels to profile. Each pair differs in exactly one property, so the
// profiler metric that explains the difference is easy to find:
//
//   1. copyCoalesced / copyStrided        global-memory coalescing (sectors per request)
//   2. transposeNaive / transposeShared /  uncoalesced stores; shared-memory bank conflicts
//      transposePadded
//   3. scaleDivergent / scaleUniform       warp divergence (branch efficiency)
//   4. polynomial                          a compute-bound kernel (FMA pipe utilisation)
//
// Build:   nvcc -O3 -lineinfo -arch=sm_80 -std=c++17 09-profile-targets.cu -o profile_targets
// NVTX:    add -DUSE_NVTX (header-only NVTX 3, ships with the CUDA toolkit)
// Profile: nsys profile --trace=cuda,nvtx ./profile_targets --bench
//          ncu --set full -k regex:transpose -o transpose ./profile_targets --bench
#include "check.cuh"

#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>
// Names a region on the Nsight Systems timeline for the lifetime of the object.
struct NvtxRange {
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
};
#else
struct NvtxRange {
    explicit NvtxRange(const char*) {}
};
#endif

constexpr int kThreads = 256;

// ---- 1. Coalescing ------------------------------------------------------------------------
// Consecutive lanes read consecutive floats: one warp request = 4 sectors of 32 bytes.
__global__ void copyCoalesced(const float* in, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

// Lane l reads element (i * stride) mod n: every lane touches a different sector, so a warp
// request needs 32 sectors and uses 4 of the 32 bytes in each. `stride` must be odd and n a
// power of two for the map to be a permutation; the store stays coalesced.
__global__ void copyStrided(const float* in, float* out, int n, int stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[static_cast<int>((static_cast<long long>(i) * stride) & (n - 1))];
}

// ---- 2. Transpose ---------------------------------------------------------------------------
// out (cols x rows) = transpose of in (rows x cols). Blocks are kTile x kRowsPerPass threads;
// each thread handles kTile / kRowsPerPass elements.
constexpr int kTile = 32;
constexpr int kRowsPerPass = 8;

// Reads are coalesced along a row of `in`; writes jump by `rows` floats between lanes.
__global__ void transposeNaive(const float* in, float* out, int rows, int cols) {
    const int x = blockIdx.x * kTile + threadIdx.x;
    for (int dy = 0; dy < kTile; dy += kRowsPerPass) {
        const int y = blockIdx.y * kTile + threadIdx.y + dy;
        if (x < cols && y < rows) out[x * rows + y] = in[y * cols + x];
    }
}

// Stage a 32 x 32 tile in shared memory so that both the read and the write are coalesced.
// kPad = 0: the column read tile[threadIdx.x][...] hits one bank 32 times (a 32-way conflict).
// kPad = 1: row r starts at bank r, so the column is spread over all 32 banks.
template <int kPad>
__global__ void transposeTiled(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[kTile][kTile + kPad];
    int x = blockIdx.x * kTile + threadIdx.x;
    for (int dy = 0; dy < kTile; dy += kRowsPerPass) {
        const int y = blockIdx.y * kTile + threadIdx.y + dy;
        if (x < cols && y < rows) tile[threadIdx.y + dy][threadIdx.x] = in[y * cols + x];
    }
    __syncthreads();
    // The block now writes the transposed tile: its x runs over the rows of `in`.
    x = blockIdx.y * kTile + threadIdx.x;
    for (int dy = 0; dy < kTile; dy += kRowsPerPass) {
        const int y = blockIdx.x * kTile + threadIdx.y + dy;
        if (x < rows && y < cols) out[y * rows + x] = tile[threadIdx.x][threadIdx.y + dy];
    }
}

// ---- 3. Divergence -------------------------------------------------------------------------
// Both kernels do the same arithmetic on the same elements; only the branch condition differs.
__device__ float slowPath(float v) {
    for (int k = 0; k < 32; ++k) v = v * 0.999f + 0.001f;
    return v;
}
__device__ float fastPath(float v) {
    for (int k = 0; k < 32; ++k) v = v * 1.001f - 0.001f;
    return v;
}

// Even and odd lanes take different paths: every warp executes both, half its lanes idle.
__global__ void scaleDivergent(const float* in, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (i % 2 == 0) ? slowPath(in[i]) : fastPath(in[i]);
}

// The condition is uniform per warp (it depends on the warp index): each warp runs one path.
// The set of elements on each path differs from scaleDivergent, but the amount of work is equal.
__global__ void scaleUniform(const float* in, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = ((i / 32) % 2 == 0) ? slowPath(in[i]) : fastPath(in[i]);
}

// ---- 4. Compute-bound --------------------------------------------------------------------
// kDegree dependent FMAs per element (Horner), 8 bytes of traffic: intensity kDegree/4 flop/B.
constexpr int kDegree = 256;
__global__ void polynomial(const float* in, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = in[i];
    float acc = 1.0f;
#pragma unroll
    for (int k = 0; k < kDegree; ++k) acc = fmaf(acc, x, 1.0f / (k + 1));
    out[i] = acc;
}

// ---- Checks --------------------------------------------------------------------------------
float slowPathHost(float v) {
    for (int k = 0; k < 32; ++k) v = v * 0.999f + 0.001f;
    return v;
}
float fastPathHost(float v) {
    for (int k = 0; k < 32; ++k) v = v * 1.001f - 0.001f;
    return v;
}

void checkCopies(int n, int stride) {
    const std::vector<float> in = ex::randomVector(n, 1);
    ex::DeviceArray<float> d_in(in), d_out(n);
    const int blocks = ex::ceilDiv(n, kThreads);

    copyCoalesced<<<blocks, kThreads>>>(d_in.ptr, d_out.ptr, n);
    CUDA_CHECK(cudaGetLastError());
    ex::check("copyCoalesced", d_out.download() == in);

    copyStrided<<<blocks, kThreads>>>(d_in.ptr, d_out.ptr, n, stride);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> ref(n);
    for (int i = 0; i < n; ++i) ref[i] = in[static_cast<int>((static_cast<long long>(i) * stride) & (n - 1))];
    ex::check("copyStrided", d_out.download() == ref);
}

void checkTransposes(int rows, int cols) {
    const std::vector<float> in = ex::randomVector(static_cast<size_t>(rows) * cols, 2);
    std::vector<float> ref(in.size());
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) ref[static_cast<size_t>(c) * rows + r] = in[static_cast<size_t>(r) * cols + c];
    ex::DeviceArray<float> d_in(in), d_out(in.size());
    const dim3 block(kTile, kRowsPerPass);
    const dim3 grid(ex::ceilDiv(cols, kTile), ex::ceilDiv(rows, kTile));
    char name[96];

    d_out.zero();
    transposeNaive<<<grid, block>>>(d_in.ptr, d_out.ptr, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    std::snprintf(name, sizeof(name), "transposeNaive %d x %d", rows, cols);
    ex::check(name, d_out.download() == ref);

    d_out.zero();
    transposeTiled<0><<<grid, block>>>(d_in.ptr, d_out.ptr, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    std::snprintf(name, sizeof(name), "transposeShared %d x %d", rows, cols);
    ex::check(name, d_out.download() == ref);

    d_out.zero();
    transposeTiled<1><<<grid, block>>>(d_in.ptr, d_out.ptr, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    std::snprintf(name, sizeof(name), "transposePadded %d x %d", rows, cols);
    ex::check(name, d_out.download() == ref);
}

void checkBranches(int n) {
    const std::vector<float> in = ex::randomVector(n, 3);
    ex::DeviceArray<float> d_in(in), d_out(n);
    const int blocks = ex::ceilDiv(n, kThreads);
    std::vector<double> ref(n);

    scaleDivergent<<<blocks, kThreads>>>(d_in.ptr, d_out.ptr, n);
    CUDA_CHECK(cudaGetLastError());
    for (int i = 0; i < n; ++i) ref[i] = (i % 2 == 0) ? slowPathHost(in[i]) : fastPathHost(in[i]);
    ex::checkClose("scaleDivergent", d_out.download(), ref);

    scaleUniform<<<blocks, kThreads>>>(d_in.ptr, d_out.ptr, n);
    CUDA_CHECK(cudaGetLastError());
    for (int i = 0; i < n; ++i) ref[i] = ((i / 32) % 2 == 0) ? slowPathHost(in[i]) : fastPathHost(in[i]);
    ex::checkClose("scaleUniform", d_out.download(), ref);
}

void checkPolynomial(int n) {
    const std::vector<float> in = ex::randomVector(n, 4, -0.5f, 0.5f);
    ex::DeviceArray<float> d_in(in), d_out(n);
    polynomial<<<ex::ceilDiv(n, kThreads), kThreads>>>(d_in.ptr, d_out.ptr, n);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> ref(n);
    for (int i = 0; i < n; ++i) {
        double acc = 1.0;
        for (int k = 0; k < kDegree; ++k) acc = acc * in[i] + 1.0 / (k + 1);
        ref[i] = acc;
    }
    ex::checkClose("polynomial", d_out.download(), ref);
}

// ---- Benchmark (the run to profile) -------------------------------------------------------
void bench() {
    const int n = 1 << 26;  // 256 MiB per array
    const std::vector<float> in = ex::randomVector(n, 5);
    ex::DeviceArray<float> d_in(in), d_out(n);
    const int blocks = ex::ceilDiv(n, kThreads);
    const double copy_bytes = 2.0 * n * sizeof(float);
    {
        NvtxRange range("copies");
        ex::reportBandwidth("copyCoalesced", ex::timeMs([&] { copyCoalesced<<<blocks, kThreads>>>(d_in.ptr, d_out.ptr, n); }),
                            copy_bytes);
        ex::reportBandwidth("copyStrided (stride 33)",
                            ex::timeMs([&] { copyStrided<<<blocks, kThreads>>>(d_in.ptr, d_out.ptr, n, 33); }), copy_bytes);
    }
    {
        NvtxRange range("transposes");
        const int side = 8192;
        const dim3 block(kTile, kRowsPerPass), grid(side / kTile, side / kTile);
        ex::reportBandwidth("transposeNaive",
                            ex::timeMs([&] { transposeNaive<<<grid, block>>>(d_in.ptr, d_out.ptr, side, side); }),
                            copy_bytes);
        ex::reportBandwidth("transposeShared",
                            ex::timeMs([&] { transposeTiled<0><<<grid, block>>>(d_in.ptr, d_out.ptr, side, side); }),
                            copy_bytes);
        ex::reportBandwidth("transposePadded",
                            ex::timeMs([&] { transposeTiled<1><<<grid, block>>>(d_in.ptr, d_out.ptr, side, side); }),
                            copy_bytes);
    }
    {
        NvtxRange range("branches");
        ex::reportBandwidth("scaleDivergent",
                            ex::timeMs([&] { scaleDivergent<<<blocks, kThreads>>>(d_in.ptr, d_out.ptr, n); }), copy_bytes);
        ex::reportBandwidth("scaleUniform",
                            ex::timeMs([&] { scaleUniform<<<blocks, kThreads>>>(d_in.ptr, d_out.ptr, n); }), copy_bytes);
    }
    {
        NvtxRange range("compute");
        const float ms = ex::timeMs([&] { polynomial<<<blocks, kThreads>>>(d_in.ptr, d_out.ptr, n); });
        std::printf("%-40s %8.3f ms  %7.1f TFLOP/s\n", "polynomial", ms, 2.0 * kDegree * n / (ms * 1e-3) * 1e-12);
    }
}

int main(int argc, char** argv) {
    for (int n : {1, 256, 4096}) checkCopies(n, 33);
    for (auto [rows, cols] : std::vector<std::pair<int, int>>{{1, 1}, {32, 32}, {40, 70}, {96, 33}}) checkTransposes(rows, cols);
    for (int n : {1, 100, 3000}) checkBranches(n);
    checkPolynomial(1000);
    if (ex::wantBench(argc, argv)) bench();
    return ex::finish("09-profile-targets");
}

// Host harness shared by the GEMM technique examples in this folder.
//
// Every example computes C = A B with A: M x K, B: K x N, C: M x N, row-major,
// FP32 output (inputs FP32, or FP16 for the tensor-core examples), and calls
// gemm::runMain() from main():
//
//   ./example              benchmark M = N = K = 4096, spot-check 256 entries
//   ./example M N K        benchmark that shape
//   ./example --test       check awkward shapes entry by entry against a CPU reference
//
// Under the CPU emulator (no GPU needed):
//   python3 tools/cuemu/cuemu.py run tutorials/gemm/01-vectorized.cu -- --test
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

namespace gemm {

__host__ __device__ inline int ceilDiv(int a, int b) { return (a + b - 1) / b; }

struct Shape {
    int m, n, k;
};

// Shapes for --test: tile multiples, ragged edges, tiny and skinny matrices.
inline std::vector<Shape> defaultTestShapes() {
    return {{128, 128, 64}, {256, 128, 40}, {132, 260, 36}, {67, 45, 33}, {1, 300, 17}, {130, 257, 96}, {200, 3, 129}};
}

// Deterministic pseudo-random values in [-1, 1).
inline float hashValue(uint32_t i, uint32_t seed) {
    uint32_t x = i * 2654435761u ^ (seed * 0x9E3779B9u + 0x7F4A7C15u);
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return static_cast<float>(x >> 8) / 8388608.0f - 1.0f;
}

inline float toFloat(float v) { return v; }
inline float toFloat(__half v) { return __half2float(v); }
template <class T> inline T fromFloat(float v);
template <> inline float fromFloat<float>(float v) { return v; }
template <> inline __half fromFloat<__half>(float v) { return __float2half(v); }

// launch(a, b, c, m, n, k): enqueue C = A B on the default stream (device pointers).
template <class T>
using LaunchFn = std::function<void(const T*, const T*, float*, int, int, int)>;

template <class T>
struct Problem {
    int m, n, k;
    std::vector<T> a, b;
    std::vector<float> c;
    T *d_a = nullptr, *d_b = nullptr;
    float* d_c = nullptr;

    Problem(int m_, int n_, int k_, uint32_t seed) : m(m_), n(n_), k(k_) {
        a.resize(static_cast<size_t>(m) * k);
        b.resize(static_cast<size_t>(k) * n);
        c.assign(static_cast<size_t>(m) * n, 0.0f);
        for (size_t i = 0; i < a.size(); ++i) a[i] = fromFloat<T>(hashValue(static_cast<uint32_t>(i), seed));
        for (size_t i = 0; i < b.size(); ++i) b[i] = fromFloat<T>(hashValue(static_cast<uint32_t>(i), seed + 1));
        CUDA_CHECK(cudaMalloc(&d_a, a.size() * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_b, b.size() * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_c, c.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_a, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice));
        // Poison the output so a kernel that skips an element cannot pass by luck.
        CUDA_CHECK(cudaMemset(d_c, 0xFF, c.size() * sizeof(float)));
    }
    ~Problem() {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
    void download() { CUDA_CHECK(cudaMemcpy(c.data(), d_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost)); }

    // Checks entry (i, j) against a double-precision dot product. The tolerance follows
    // the standard bound for a length-K dot product in FP32: |error| <= K u sum |a b|.
    bool checkEntry(int i, int j) const {
        double ref = 0.0, mag = 0.0;
        for (int kk = 0; kk < k; ++kk) {
            const double p = static_cast<double>(toFloat(a[static_cast<size_t>(i) * k + kk])) *
                             toFloat(b[static_cast<size_t>(kk) * n + j]);
            ref += p;
            mag += std::fabs(p);
        }
        const double got = c[static_cast<size_t>(i) * n + j];
        const double tol = 2.0 * k * 5.96e-8 * mag + 1e-6;
        if (!(std::fabs(got - ref) <= tol)) {
            std::printf("  mismatch at (%d, %d): got %.7g, expected %.7g\n", i, j, got, ref);
            return false;
        }
        return true;
    }
};

template <class T>
bool testShape(const LaunchFn<T>& launch, const Shape& s) {
    Problem<T> p(s.m, s.n, s.k, 1234u + s.m * 7u + s.n * 13u + s.k);
    launch(p.d_a, p.d_b, p.d_c, s.m, s.n, s.k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    p.download();
    int bad = 0;
    for (int i = 0; i < s.m && bad < 5; ++i)
        for (int j = 0; j < s.n && bad < 5; ++j) bad += !p.checkEntry(i, j);
    std::printf("%s M=%d N=%d K=%d\n", bad ? "FAIL" : "ok  ", s.m, s.n, s.k);
    return bad == 0;
}

template <class T>
int runMain(const char* name, const LaunchFn<T>& launch, int argc, char** argv,
            std::vector<Shape> test_shapes = defaultTestShapes()) {
    if (argc > 1 && std::strcmp(argv[1], "--test") == 0) {
        bool ok = true;
        for (const Shape& s : test_shapes) ok &= testShape(launch, s);
        std::printf("%s: %s\n", name, ok ? "all shapes passed" : "FAILED");
        return ok ? 0 : 1;
    }
    Shape s{4096, 4096, 4096};
    if (argc == 4) s = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};
    Problem<T> p(s.m, s.n, s.k, 42u);

    launch(p.d_a, p.d_b, p.d_c, s.m, s.n, s.k);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    constexpr int kReps = 10;
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventRecord(start_event));
    for (int rep = 0; rep < kReps; ++rep) launch(p.d_a, p.d_b, p.d_c, s.m, s.n, s.k);
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    const double seconds = elapsed_ms * 1e-3 / kReps;
    const double flops = 2.0 * s.m * s.n * s.k;

    // Spot-check 256 entries spread over the output (a full CPU check would take minutes).
    p.download();
    int bad = 0;
    for (int t = 0; t < 256; ++t) {
        const int i = static_cast<int>((static_cast<uint64_t>(t) * 2654435761u) % s.m);
        const int j = static_cast<int>((static_cast<uint64_t>(t) * 40503u + 17u) % s.n);
        bad += !p.checkEntry(i, j);
    }
    std::printf("%s  M=%d N=%d K=%d  %.3f ms  %.2f TFLOP/s  check %s\n", name, s.m, s.n, s.k, seconds * 1e3,
                seconds > 0 ? flops / seconds * 1e-12 : 0.0, bad ? "FAILED" : "ok");
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    return bad ? 1 : 0;
}

}  // namespace gemm

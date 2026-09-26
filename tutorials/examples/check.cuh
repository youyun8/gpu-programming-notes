// Small helpers shared by the example programs of chapters 09-13.
//
// Every program checks its kernels against a CPU reference and prints one line
// per check; with --bench it also times them on a larger input:
//
//   ./example            run the checks (the default, and what CI does on cuemu)
//   ./example --bench    run the checks, then time the kernels (needs a real GPU)
//
// Without a GPU: python3 tools/cuemu/cuemu.py run tutorials/examples/11-scan.cu
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err_ = (call);                                                           \
        if (err_ != cudaSuccess) {                                                           \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

namespace ex {

__host__ __device__ inline int ceilDiv(int a, int b) { return (a + b - 1) / b; }

// Deterministic pseudo-random numbers in [lo, hi).
inline float randomValue(uint32_t i, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    uint32_t x = i * 2654435761u ^ (seed * 0x9E3779B9u + 0x7F4A7C15u);
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return lo + (hi - lo) * static_cast<float>(x >> 8) / 16777216.0f;
}

inline std::vector<float> randomVector(size_t n, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = randomValue(static_cast<uint32_t>(i), seed, lo, hi);
    return v;
}

// A device copy of a host vector (and back).
template <class T>
struct DeviceArray {
    T* ptr = nullptr;
    size_t count = 0;
    explicit DeviceArray(size_t n) : count(n) { CUDA_CHECK(cudaMalloc(&ptr, std::max<size_t>(n, 1) * sizeof(T))); }
    explicit DeviceArray(const std::vector<T>& host) : DeviceArray(host.size()) { upload(host); }
    ~DeviceArray() { cudaFree(ptr); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    void upload(const std::vector<T>& host) {
        CUDA_CHECK(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> host(count);
        CUDA_CHECK(cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }
    void zero() { CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T))); }
};

// Largest |got - ref| / (atol + rtol |ref|) over the arrays; <= 1 means "close".
inline double errorRatio(const std::vector<float>& got, const std::vector<double>& ref, double rtol, double atol) {
    if (got.size() != ref.size()) return 1e30;
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double r = std::fabs(got[i] - ref[i]) / (atol + rtol * std::fabs(ref[i]));
        if (!(r <= worst)) worst = r;  // also catches NaN
    }
    return worst;
}

inline int& failures() {
    static int count = 0;
    return count;
}

// Prints and records one check.
inline bool check(const char* what, bool ok) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures();
    return ok;
}

inline bool checkClose(const char* what, const std::vector<float>& got, const std::vector<double>& ref,
                       double rtol = 1e-4, double atol = 1e-5) {
    const double e = errorRatio(got, ref, rtol, atol);
    char line[256];
    std::snprintf(line, sizeof(line), "%s (error ratio %.3g)", what, e);
    return check(line, e <= 1.0);
}

// Average time of launch() in milliseconds, after one warm-up call.
inline float timeMs(const std::function<void()>& launch, int reps = 20) {
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventRecord(start_event));
    for (int r = 0; r < reps; ++r) launch();
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    return ms / reps;
}

// Prints time and effective bandwidth for `bytes` moved per call.
inline void reportBandwidth(const char* what, float ms, double bytes) {
    std::printf("%-40s %8.3f ms  %7.1f GB/s\n", what, ms, ms > 0 ? bytes / (ms * 1e-3) * 1e-9 : 0.0);
}

inline bool wantBench(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--bench") == 0) return true;
    return false;
}

inline int finish(const char* program) {
    std::printf("%s: %s\n", program, failures() ? "FAILED" : "all checks passed");
    return failures() ? 1 : 0;
}

}  // namespace ex

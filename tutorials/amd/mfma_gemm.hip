// Teaching kernel for tutorial 05: a bf16 "TN" GEMM on CDNA3 (MI300, gfx942)
// built from v_mfma_f32_16x16x16_bf16.
//
//   C[M][N] (fp32) = A[M][K] (bf16, row-major) * B[N][K]^T (bf16, row-major)
//
// This is the same operand layout AITER's asm kernels use ("tn"): both A and B
// are K-contiguous, so every MFMA operand is 4 consecutive bf16 = one 8-byte
// LDS read per lane.
//
// Block tile 64x64x32, 256 threads = 4 wave64s arranged 2x2, each wave owns a
// 32x32 sub-tile = 2x2 MFMA tiles of 16x16. LDS is double buffered so a
// single barrier per K step is enough. M, N must be multiples of 64 and K a
// multiple of 32 (checked on the host).
//
// Build (ROCm):   hipcc -O3 --offload-arch=gfx942 mfma_gemm.hip -o mfma_gemm
// ISA only:       see hip_compat.h
#include "hip_compat.h"

#include <stdint.h>

typedef float Float4 __attribute__((ext_vector_type(4)));
typedef short Short4 __attribute__((ext_vector_type(4)));
typedef unsigned int Uint4 __attribute__((ext_vector_type(4)));

constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kBlockK = 32;
constexpr int kThreads = 256;
constexpr int kWaveSize = 64;
// Pad each LDS row by 8 bf16 (16 bytes): row stride 80 bytes spreads the
// 16 rows touched by one MFMA operand read over different banks.
constexpr int kLdsStride = kBlockK + 8;

struct __attribute__((aligned(16))) SharedTiles {
  uint16_t a[2][kBlockM * kLdsStride];
  uint16_t b[2][kBlockN * kLdsStride];
};

// Global -> registers: 64 rows x 32 bf16 = 4 KiB per operand = 16 bytes per thread.
__device__ inline Uint4 loadTile(const uint16_t* base, int ld, int thread) {
  int row = thread / 4;
  int col = (thread % 4) * 8;
  return *reinterpret_cast<const Uint4*>(base + static_cast<int64_t>(row) * ld + col);
}

// Registers -> LDS, same (row, col) mapping as loadTile.
__device__ inline void storeTile(uint16_t* tile, Uint4 value, int thread) {
  int row = thread / 4;
  int col = (thread % 4) * 8;
  *reinterpret_cast<Uint4*>(tile + row * kLdsStride + col) = value;
}

// One MFMA operand: lane l supplies row (l % 16), k = k0 + 4 * (l / 16) .. +3.
__device__ inline Short4 readOperand(const uint16_t* tile, int row0, int k0, int lane) {
  const uint16_t* p = tile + (row0 + lane % 16) * kLdsStride + k0 + 4 * (lane / 16);
  return *reinterpret_cast<const Short4*>(p);
}

extern "C" __global__ __launch_bounds__(kThreads) void mfmaGemmTn(const uint16_t* a, const uint16_t* b, float* c,
                                                                  int m, int n, int k) {
  __shared__ SharedTiles smem;

  int thread = threadIdx_x;
  int lane = thread % kWaveSize;
  int wave = thread / kWaveSize;
  int wave_m = (wave / 2) * 32;
  int wave_n = (wave % 2) * 32;
  int block_m = blockIdx_y * kBlockM;
  int block_n = blockIdx_x * kBlockN;

  const uint16_t* a_block = a + static_cast<int64_t>(block_m) * k;
  const uint16_t* b_block = b + static_cast<int64_t>(block_n) * k;

  Float4 acc[2][2] = {};

  // Prologue: stage K tile 0.
  storeTile(smem.a[0], loadTile(a_block, k, thread), thread);
  storeTile(smem.b[0], loadTile(b_block, k, thread), thread);
  __syncthreads();

  int num_k_tiles = k / kBlockK;
  for (int kt = 0; kt < num_k_tiles; ++kt) {
    int cur = kt & 1;
    bool has_next = kt + 1 < num_k_tiles;

    // Issue the next tile's global loads first; their latency hides behind the MFMAs.
    Uint4 next_a = {}, next_b = {};
    if (has_next) {
      next_a = loadTile(a_block + (kt + 1) * kBlockK, k, thread);
      next_b = loadTile(b_block + (kt + 1) * kBlockK, k, thread);
    }

#pragma unroll
    for (int k0 = 0; k0 < kBlockK; k0 += 16) {
      Short4 a_frag[2], b_frag[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        a_frag[i] = readOperand(smem.a[cur], wave_m + i * 16, k0, lane);
        b_frag[i] = readOperand(smem.b[cur], wave_n + i * 16, k0, lane);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag[i], b_frag[j], acc[i][j], 0, 0, 0);
        }
      }
    }

    // The other buffer was last read in iteration kt - 1, which ended with a
    // barrier, so it is free to overwrite now.
    if (has_next) {
      storeTile(smem.a[cur ^ 1], next_a, thread);
      storeTile(smem.b[cur ^ 1], next_b, thread);
    }
    __syncthreads();
  }

  // Accumulator layout of a 16x16 MFMA: lane l holds rows 4 * (l / 16) + r, column l % 16.
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int col = block_n + wave_n + j * 16 + lane % 16;
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        int row = block_m + wave_m + i * 16 + 4 * (lane / 16) + r;
        c[static_cast<int64_t>(row) * n + col] = acc[i][j][r];
      }
    }
  }
}

#ifndef HIP_COMPAT_DEVICE_ONLY
// ---------------------------------------------------------------------------------------------
// Host driver: random test against a CPU reference, then a simple timing loop.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define HIP_CHECK(expr)                                                              \
  do {                                                                               \
    hipError_t err_ = (expr);                                                        \
    if (err_ != hipSuccess) {                                                        \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(err_)); \
      std::exit(1);                                                                  \
    }                                                                                \
  } while (0)

static uint16_t floatToBf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  bits += 0x7FFF + ((bits >> 16) & 1);  // round to nearest even
  return static_cast<uint16_t>(bits >> 16);
}

static float bf16ToFloat(uint16_t h) {
  uint32_t bits = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

int main(int argc, char** argv) {
  int m = argc > 1 ? std::atoi(argv[1]) : 1024;
  int n = argc > 2 ? std::atoi(argv[2]) : 1024;
  int k = argc > 3 ? std::atoi(argv[3]) : 1024;
  if (m % kBlockM || n % kBlockN || k % kBlockK) {
    std::fprintf(stderr, "M, N must be multiples of 64 and K a multiple of 32\n");
    return 1;
  }

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<uint16_t> h_a(static_cast<size_t>(m) * k), h_b(static_cast<size_t>(n) * k);
  for (auto& v : h_a) v = floatToBf16(dist(rng));
  for (auto& v : h_b) v = floatToBf16(dist(rng));

  uint16_t *d_a, *d_b;
  float* d_c;
  HIP_CHECK(hipMalloc(&d_a, h_a.size() * 2));
  HIP_CHECK(hipMalloc(&d_b, h_b.size() * 2));
  HIP_CHECK(hipMalloc(&d_c, static_cast<size_t>(m) * n * 4));
  HIP_CHECK(hipMemcpy(d_a, h_a.data(), h_a.size() * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), h_b.size() * 2, hipMemcpyHostToDevice));

  dim3 grid(n / kBlockN, m / kBlockM);
  mfmaGemmTn<<<grid, kThreads>>>(d_a, d_b, d_c, m, n, k);
  HIP_CHECK(hipGetLastError());
  std::vector<float> h_c(static_cast<size_t>(m) * n);
  HIP_CHECK(hipMemcpy(h_c.data(), d_c, h_c.size() * 4, hipMemcpyDeviceToHost));

  double max_err = 0;
  for (int i = 0; i < m; i += 7) {
    for (int j = 0; j < n; j += 5) {
      double ref = 0;
      for (int p = 0; p < k; ++p) {
        ref += static_cast<double>(bf16ToFloat(h_a[static_cast<size_t>(i) * k + p])) *
               bf16ToFloat(h_b[static_cast<size_t>(j) * k + p]);
      }
      max_err = std::fmax(max_err, std::fabs(ref - h_c[static_cast<size_t>(i) * n + j]));
    }
  }
  std::printf("max abs error vs fp64 reference: %.3e\n", max_err);

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  constexpr int kIters = 20;
  HIP_CHECK(hipEventRecord(start));
  for (int it = 0; it < kIters; ++it) mfmaGemmTn<<<grid, kThreads>>>(d_a, d_b, d_c, m, n, k);
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));
  float ms = 0;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  double tflops = 2.0 * m * n * k * kIters / (ms * 1e-3) / 1e12;
  std::printf("%d x %d x %d: %.3f ms/iter, %.1f TFLOP/s\n", m, n, k, ms / kIters, tflops);

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));
  return max_err < 1e-2 * std::sqrt(static_cast<double>(k)) ? 0 : 1;
}
#endif

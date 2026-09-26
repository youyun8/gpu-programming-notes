// GEMM technique 3: asynchronous global -> shared copies (cp.async, sm_80+)
// with a multi-stage pipeline.
//
// cp.async copies 4, 8 or 16 bytes from global to shared memory without passing
// through registers, and without blocking the thread: it is only waited for, per
// *group* of copies, with cp.async.wait_group. Here it is used through the
// <cuda_pipeline.h> primitives, which compile to exactly those instructions:
//
//   __pipeline_memcpy_async(dst, src, 16)  cp.async.{ca,cg}.shared.global [dst], [src], 16
//   __pipeline_commit()                    cp.async.commit_group
//   __pipeline_wait_prior(n)               cp.async.wait_group n  (all but the n newest groups done)
//
// With kStages buffers, kStages - 1 slices are in flight while one is computed:
//
//   prologue: issue slices 0 .. kStages-2, one group each
//   step s:   wait until slice s has landed (wait_prior(kStages - 2)); barrier
//             issue slice s + kStages - 1 into the buffer that held slice s - 1
//             compute slice s
//
// cp.async cannot transpose, so A is kept row-major in shared memory and read with
// scalar loads (lanes of a warp share 2 row groups: broadcasts). Tile shapes and the
// thread mapping of the math are those of 01-vectorized.cu.
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 03-cp-async.cu -o cp_async
#include <cuda_pipeline.h>

#include "harness.cuh"

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 8;
constexpr int kThreads = 256;
constexpr int kStages = 3;
constexpr int kStrideA = kBlockK + 4;  // 12 floats = 48 bytes: rows stay 16-byte aligned, and
                                       // rows 4 apart start 16 banks apart (conflict-free reads)

// Issues the copies of one K slice into stage `stage`. kVec = 4: one 16-byte copy of A
// and one of B per thread (needs K % 4 == 0 and N % 4 == 0); kVec = 1: 4-byte copies.
// Elements outside the matrices are zero-filled by the copy itself (src-size 0).
template <int kVec>
__device__ __forceinline__ void issueSlice(float (*a_s)[kStrideA], float (*b_s)[kBlockN], const float* a,
                                           const float* b, int m, int n, int k, int row0, int col0, int k0) {
    constexpr int kBytes = 4 * kVec;
    constexpr int kCopies = 4 / kVec;  // copies per thread per operand
    const int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < kCopies; ++i) {
        // A slice: 128 rows x 8 floats. B slice: 8 rows x 128 floats. Each copy moves kVec floats.
        const int a_idx = (tid * kCopies + i) * kVec;
        const int a_r = a_idx / kBlockK, a_c = a_idx % kBlockK;
        const bool a_in = row0 + a_r < m && k0 + a_c < k;
        const float* a_src = a_in ? a + static_cast<size_t>(row0 + a_r) * k + k0 + a_c : a;
        __pipeline_memcpy_async(&a_s[a_r][a_c], a_src, kBytes, a_in ? 0 : kBytes);

        const int b_idx = (tid * kCopies + i) * kVec;
        const int b_r = b_idx / kBlockN, b_c = b_idx % kBlockN;
        const bool b_in = k0 + b_r < k && col0 + b_c < n;
        const float* b_src = b_in ? b + static_cast<size_t>(k0 + b_r) * n + col0 + b_c : b;
        __pipeline_memcpy_async(&b_s[b_r][b_c], b_src, kBytes, b_in ? 0 : kBytes);
    }
}

template <int kVec>
__global__ void __launch_bounds__(kThreads) sgemmCpAsync(const float* __restrict__ a, const float* __restrict__ b,
                                                         float* __restrict__ c, int m, int n, int k) {
    __shared__ __align__(16) float a_s[kStages][kBlockM][kStrideA];
    __shared__ __align__(16) float b_s[kStages][kBlockK][kBlockN];

    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int row0 = blockIdx.y * kBlockM;
    const int col0 = blockIdx.x * kBlockN;
    const int num_slices = (k + kBlockK - 1) / kBlockK;

    // Prologue: kStages - 1 slices in flight. A group is committed even when it is empty,
    // so that "group s holds slice s" stays true and the wait count below is a constant.
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < num_slices) issueSlice<kVec>(a_s[s], b_s[s], a, b, m, n, k, row0, col0, s * kBlockK);
        __pipeline_commit();
    }

    float acc[8][8] = {};
    for (int s = 0; s < num_slices; ++s) {
        // Groups 0 .. s + kStages - 2 are committed; allowing kStages - 2 of them to be
        // pending means groups 0 .. s have completed: slice s is in shared memory
        // (for this thread's copies; the barrier extends that to the whole block).
        __pipeline_wait_prior(kStages - 2);
        __syncthreads();

        // Refill the stage that held slice s - 1: every thread has passed the barrier
        // above, so every thread has finished computing on it.
        const int next = s + kStages - 1;
        if (next < num_slices) {
            const int st = next % kStages;
            issueSlice<kVec>(a_s[st], b_s[st], a, b, m, n, k, row0, col0, next * kBlockK);
        }
        __pipeline_commit();

        const int st = s % kStages;
#pragma unroll
        for (int kk = 0; kk < kBlockK; ++kk) {
            float a_frag[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a_frag[i] = a_s[st][4 * ty + i][kk];
                a_frag[4 + i] = a_s[st][64 + 4 * ty + i][kk];
            }
            const float4 b_lo = *reinterpret_cast<const float4*>(&b_s[st][kk][4 * tx]);
            const float4 b_hi = *reinterpret_cast<const float4*>(&b_s[st][kk][64 + 4 * tx]);
            const float b_frag[8] = {b_lo.x, b_lo.y, b_lo.z, b_lo.w, b_hi.x, b_hi.y, b_hi.z, b_hi.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
        }
    }
    __pipeline_wait_prior(0);  // do not exit with copies in flight

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row0 + (i < 4 ? 4 * ty + i : 64 + 4 * ty + (i - 4));
        if (row >= m) continue;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int col = col0 + 64 * half + 4 * tx;
            float* dst = c + static_cast<size_t>(row) * n + col;
            const float* v = &acc[i][4 * half];
            if (kVec == 4) {
                if (col < n) *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (col + j < n) dst[j] = v[j];
            }
        }
    }
}

void launchCpAsync(const float* a, const float* b, float* c, int m, int n, int k) {
    const dim3 grid(gemm::ceilDiv(n, kBlockN), gemm::ceilDiv(m, kBlockM));
    if (k % 4 == 0 && n % 4 == 0)
        sgemmCpAsync<4><<<grid, kThreads>>>(a, b, c, m, n, k);
    else
        sgemmCpAsync<1><<<grid, kThreads>>>(a, b, c, m, n, k);
}

int main(int argc, char** argv) {
    return gemm::runMain<float>("03-cp-async", launchCpAsync, argc, argv);
}

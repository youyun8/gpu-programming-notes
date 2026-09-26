// Max Subarray Sum (LeetGPU)
// https://leetgpu.com/challenges/max-subarray-sum
//
// Max over all windows of length w of the window sum. With prefix sums P,
// window(i) = P[i + w] - P[i], so the whole problem is one scan + one max.
// N <= 50,000 fits comfortably in a single 1024-thread block: the block scans
// the input in 1024-element chunks into a global prefix array (carrying the
// running total), then every thread takes the max over its windows.
#include <cuda_runtime.h>
#include <climits>

constexpr int kThreads = 1024;

__device__ int blockInclusiveScan(int v, int* warp_totals) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, v, offset);
        if (lane >= offset) v += other;
    }
    if (lane == 31) warp_totals[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int t = warp_totals[lane];
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int other = __shfl_up_sync(0xffffffffu, t, offset);
            if (lane >= offset) t += other;
        }
        warp_totals[lane] = t;
    }
    __syncthreads();
    if (warp > 0) v += warp_totals[warp - 1];
    return v;
}

__global__ void maxWindowSum(const int* input, int* prefix, int* output, int n, int w) {
    __shared__ int warp_totals[32];
    __shared__ int carry;
    __shared__ int warp_max[32];
    if (threadIdx.x == 0) {
        carry = 0;
        prefix[0] = 0;
    }
    __syncthreads();
    for (int base = 0; base < n; base += kThreads) {
        const int i = base + threadIdx.x;
        const int v = i < n ? input[i] : 0;
        const int inclusive = blockInclusiveScan(v, warp_totals) + carry;
        if (i < n) prefix[i + 1] = inclusive;
        __syncthreads();
        if (threadIdx.x == kThreads - 1) carry = inclusive;
        __syncthreads();
    }
    int best = INT_MIN;
    for (int i = threadIdx.x; i + w <= n; i += kThreads) best = max(best, prefix[i + w] - prefix[i]);
    for (int offset = 16; offset > 0; offset >>= 1) best = max(best, __shfl_xor_sync(0xffffffffu, best, offset));
    if (threadIdx.x % 32 == 0) warp_max[threadIdx.x / 32] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
        int m = warp_max[0];
        for (int wi = 1; wi < kThreads / 32; ++wi) m = max(m, warp_max[wi]);
        output[0] = m;
    }
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    int* prefix = nullptr;
    cudaMalloc(&prefix, (N + 1) * sizeof(int));
    maxWindowSum<<<1, kThreads>>>(input, prefix, output, N, window_size);
    cudaDeviceSynchronize();
    cudaFree(prefix);
}

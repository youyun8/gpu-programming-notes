// Parallel Merge (LeetGPU)
// https://leetgpu.com/challenges/parallel-merge
//
// Merge path: output position k of the merge takes i elements from A and
// k - i from B, where i (the "co-rank") is found by a binary search on the
// cross diagonal: the smallest i with A[i] > B[k - i - 1]. Every thread
// co-ranks the start of its 8-element output segment independently, then
// merges its segment sequentially. No synchronization, O(log n) setup per thread.
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kItemsPerThread = 8;

// Number of elements taken from a for the first k outputs (ties favour a).
__device__ int coRank(int k, const float* a, int m, const float* b, int n) {
    int lo = k > n ? k - n : 0;
    int hi = k < m ? k : m;
    while (lo < hi) {
        const int i = (lo + hi) / 2;  // candidate: take i from a, k - i from b
        const int j = k - i;
        if (a[i] <= b[j - 1]) {
            lo = i + 1;  // a[i] must come before b[j - 1]: take more from a
        } else {
            hi = i;
        }
    }
    return lo;
}

__global__ void mergePath(const float* a, const float* b, float* c, int m, int n) {
    const long long total = static_cast<long long>(m) + n;
    const long long start = (blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x) * kItemsPerThread;
    if (start >= total) return;
    const int k = static_cast<int>(start);
    int i = coRank(k, a, m, b, n);
    int j = k - i;
    const int end = static_cast<int>(start + kItemsPerThread < total ? start + kItemsPerThread : total);
    for (int out = k; out < end; ++out) {
        if (j >= n || (i < m && a[i] <= b[j])) {
            c[out] = a[i++];
        } else {
            c[out] = b[j++];
        }
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    const long long threads = (static_cast<long long>(M) + N + kItemsPerThread - 1) / kItemsPerThread;
    const int blocks = static_cast<int>((threads + kBlockSize - 1) / kBlockSize);
    mergePath<<<blocks, kBlockSize>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}

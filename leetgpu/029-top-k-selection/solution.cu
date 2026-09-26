// Top-K Selection (LeetGPU)
// https://leetgpu.com/challenges/top-k-selection
//
// 1. Radix select: floats are mapped to order-preserving uint32 keys; four
//    8-bit passes (histogram over the candidates that match the prefix found
//    so far) pin down the exact key T of the k-th largest element and how
//    many copies of T belong to the answer. Everything stays on the device.
// 2. Gather: every element with key > T is appended to a small buffer; the
//    remaining slots are filled with T.
// 3. Sort the k survivors descending with a bitonic sort (one block in shared
//    memory when k <= 2048, global-memory passes otherwise).
// Cost: ~5 streaming passes over the input, independent of k.
#include <cuda_runtime.h>

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 1024;
constexpr int kSmemSort = 2048;

// Radix select of the k-th largest key T, one 8-bit digit at a time (most significant first).
struct SelectState {
    unsigned int prefix;     // high bits of T found so far
    unsigned int mask;       // which bits of prefix are valid
    unsigned int remaining;  // rank of T among the candidates matching prefix
    unsigned int count_gt;   // gather cursor for keys > T
};

__device__ SelectState g_state;
__device__ unsigned int g_hist[256];

// Map floats to unsigned keys whose integer order equals the float order
// (flip all bits of negatives, set the sign bit of positives).
__device__ __forceinline__ unsigned int floatToKey(float f) {
    const unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float keyToFloat(unsigned int k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k);
}

__global__ void initState(int k) {
    g_state = SelectState{0u, 0u, static_cast<unsigned int>(k), 0u};
}

// Histogram of the current digit over the keys that still match the prefix found so far.
__global__ void digitHistogram(const float* input, int n, int shift) {
    __shared__ unsigned int s_hist[256];
    s_hist[threadIdx.x] = 0;
    __syncthreads();
    const unsigned int prefix = g_state.prefix;
    const unsigned int mask = g_state.mask;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const unsigned int key = floatToKey(input[i]);
        if ((key & mask) == prefix) atomicAdd(&s_hist[(key >> shift) & 0xFFu], 1u);
    }
    __syncthreads();
    if (s_hist[threadIdx.x]) atomicAdd(&g_hist[threadIdx.x], s_hist[threadIdx.x]);
}

// One thread walks the 256 buckets from the largest digit down.
__global__ void chooseDigit(int shift) {
    unsigned int remaining = g_state.remaining;
    int digit = 255;
    for (; digit > 0; --digit) {
        const unsigned int count = g_hist[digit];
        if (count >= remaining) break;
        remaining -= count;
    }
    g_state.prefix |= static_cast<unsigned int>(digit) << shift;
    g_state.mask |= 0xFFu << shift;
    g_state.remaining = remaining;
    for (int i = 0; i < 256; ++i) g_hist[i] = 0;
}

// Collect every key strictly greater than T (order does not matter yet).
__global__ void gatherGreater(const float* input, int n, unsigned int* selected) {
    const unsigned int t = g_state.prefix;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const unsigned int key = floatToKey(input[i]);
        if (key > t) selected[atomicAdd(&g_state.count_gt, 1u)] = key;
    }
}

// Slots [count_gt, k) hold copies of T; [k, padded) hold 0 (sorts last).
__global__ void fillTail(unsigned int* selected, int k, int padded) {
    const unsigned int start = g_state.count_gt;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < padded; i += gridDim.x * blockDim.x) {
        if (i >= static_cast<int>(start)) selected[i] = i < k ? g_state.prefix : 0u;
    }
}

// Descending bitonic sort of up to 2048 keys in one block.
__global__ void bitonicSmem(unsigned int* keys, int padded) {
    __shared__ unsigned int s[kSmemSort];
    for (int i = threadIdx.x; i < padded; i += blockDim.x) s[i] = keys[i];
    __syncthreads();
    for (int size = 2; size <= padded; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < padded; i += blockDim.x) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool descending = (i & size) == 0;
                    const unsigned int a = s[i];
                    const unsigned int b = s[j];
                    if ((a < b) == descending) {
                        s[i] = b;
                        s[j] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < padded; i += blockDim.x) keys[i] = s[i];
}

// One compare-exchange step of a global bitonic sort (for more than 2048 keys).
__global__ void bitonicStep(unsigned int* keys, int padded, int size, int stride) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < padded; i += gridDim.x * blockDim.x) {
        const int j = i ^ stride;
        if (j > i) {
            const bool descending = (i & size) == 0;
            const unsigned int a = keys[i];
            const unsigned int b = keys[j];
            if ((a < b) == descending) {
                keys[i] = b;
                keys[j] = a;
            }
        }
    }
}

// Convert the first k sorted keys back to floats.
__global__ void writeOutput(const unsigned int* keys, float* output, int k) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) output[i] = keyToFloat(keys[i]);
}

static int gridFor(long long work) {
    long long blocks = (work + kBlockSize - 1) / kBlockSize;
    return static_cast<int>(blocks < 1 ? 1 : (blocks > kMaxBlocks ? kMaxBlocks : blocks));
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    // Phase 1: four digit passes find T exactly (histogram, then a one-thread digit choice).
    initState<<<1, 1>>>(k);
    const int grid = gridFor(N);
    for (int shift = 24; shift >= 0; shift -= 8) {
        digitHistogram<<<grid, 256>>>(input, N, shift);
        chooseDigit<<<1, 1>>>(shift);
    }

    // Phase 2: gather keys > T, pad with copies of T up to k (ties) and zeros up to a power of two.
    int padded = 1;
    while (padded < k) padded <<= 1;
    unsigned int* selected = nullptr;
    cudaMalloc(&selected, padded * sizeof(unsigned int));
    gatherGreater<<<grid, kBlockSize>>>(input, N, selected);
    fillTail<<<gridFor(padded), kBlockSize>>>(selected, k, padded);

    // Phase 3: sort descending, in one block's shared memory when it fits.
    if (padded <= kSmemSort) {
        bitonicSmem<<<1, 1024>>>(selected, padded);
    } else {
        for (int size = 2; size <= padded; size <<= 1)
            for (int stride = size >> 1; stride > 0; stride >>= 1)
                bitonicStep<<<gridFor(padded), kBlockSize>>>(selected, padded, size, stride);
    }
    writeOutput<<<gridFor(k), kBlockSize>>>(selected, output, k);
    cudaDeviceSynchronize();
    cudaFree(selected);
}

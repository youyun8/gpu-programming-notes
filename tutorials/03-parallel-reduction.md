# 03 – Parallel Reduction

Goal: `sum = Σ input[i]`. The pattern generalizes to max, argmax, dot product,
softmax denominators, norms…

## Step 1 – Block-level tree reduction in shared memory

```cpp
constexpr int kBlockSize = 256;

__global__ void reduceSum(const float* input, float* output, int n) {
    __shared__ float cache[kBlockSize];
    const int tid = threadIdx.x;

    // Grid-stride accumulate into a register first: fewer blocks, less atomics.
    float local_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) {
        local_sum += input[i];
    }
    cache[tid] = local_sum;
    __syncthreads();

    // Sequential addressing: active threads stay contiguous -> no divergence
    // within full warps, no bank conflicts.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) cache[tid] += cache[tid + stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, cache[0]);
}
```

`*output` must be zeroed before the launch (`cudaMemset`).

## Step 2 – Warp shuffles

Within a warp no shared memory or `__syncthreads()` is needed:

```cpp
__device__ float warpReduceSum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;   // valid in lane 0
}

__device__ float blockReduceSum(float value) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    value = warpReduceSum(value);
    if (lane == 0) warp_sums[warp_id] = value;
    __syncthreads();

    const int num_warps = (blockDim.x + 31) / 32;
    value = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.0f;
    if (warp_id == 0) value = warpReduceSum(value);
    return value;   // valid in thread 0
}
```

## Evolution (Mark Harris' classic)

1. Interleaved addressing with `tid % (2*s) == 0` → heavy divergence.
2. Interleaved with strided index → bank conflicts.
3. Sequential addressing (above).
4. First add during global load (halves idle threads).
5. Unroll the last warp → today: warp shuffles.
6. Multiple elements per thread (grid-stride) → bandwidth-bound optimum.

## Precision

Float addition is not associative: GPU results differ slightly from a
sequential CPU sum. Judges use a tolerance; for large `n` consider Kahan
summation or accumulating in `double`.

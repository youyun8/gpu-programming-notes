# 03 – Parallel Reduction

Goal: $s = \sum_i x_i$. The same pattern computes maxima, arg-max, dot
products, norms, softmax denominators, means and variances. This chapter
covers:

- the work–depth view of a parallel reduction;
- block reductions, first in shared memory and then with warp shuffles;
- how to finish across blocks;
- floating-point accuracy;
- the general "reduce with a monoid" recipe that the problem pages reuse.

## 1. Work, Depth and Brent's Bound

A sequential sum does $n - 1$ additions in a chain of length $n - 1$. A
balanced tree does the same additions in far fewer levels:

$$
W = n - 1, \qquad D = \lceil \log_2 n \rceil, \qquad
T_p \le \frac{W}{p} + D
$$

| Symbol | Meaning |
|---|---|
| $n$ | Number of inputs |
| $W$ | Work: total number of additions |
| $D$ | Depth: length of the longest chain of dependent additions |
| $p$ | Number of processors (threads working at once) |
| $T_p$ | Time steps with $p$ processors (Brent's theorem) |

The tree is **work-efficient** (same $W$ as the sequential sum) and has
logarithmic depth. On a GPU, $p$ is in the tens of thousands while $n$ is in
the millions, so the $W/p$ term dominates. The practical recipe follows
from that: each thread first sums many elements **sequentially** (cheap,
no synchronization), and only the few per-thread results go through a tree.

## 2. Step 1: Block-Level Tree Reduction in Shared Memory

```cpp
constexpr int kBlockSize = 256;

__global__ void reduceSum(const float* input, float* output, int n) {
    __shared__ float cache[kBlockSize];
    const int tid = threadIdx.x;

    // Grid-stride accumulation in a register first: fewer blocks, fewer atomics.
    float local_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) {
        local_sum += input[i];
    }
    cache[tid] = local_sum;
    __syncthreads();

    // Sequential addressing: active threads stay contiguous, so full warps
    // either all work or all idle (no divergence) and there are no bank conflicts.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) cache[tid] += cache[tid + stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, cache[0]);
}
```

`*output` must be zeroed before the launch (`cudaMemset`). The tree has
$\log_2 256 = 8$ levels, each ending in a barrier.

![Sequential addressing: at each level the first half of the active threads adds the second half](figures/ch03-tree.svg)

## 3. Step 2: Warp Shuffles

Within a warp no shared memory or barrier is needed; `__shfl_down_sync`
reads a register of another lane:

$$
v^{(s+1)}_\ell = v^{(s)}_\ell + v^{(s)}_{\ell + \delta_s}, \qquad \delta_s = 16, 8, 4, 2, 1
$$

| Symbol | Meaning |
|---|---|
| $\ell$ | Lane index |
| $v^{(s)}_\ell$ | Lane $\ell$'s value after step $s$ |
| $\delta_s$ | Shuffle offset at step $s$; after 5 steps lane 0 holds the warp's sum |

![__shfl_down_sync with offsets 8, 4, 2, 1: after log2(width) steps lane 0 holds the sum](figures/ch03-shuffle.svg)

With `__shfl_xor_sync` (a butterfly) instead, *every* lane ends with the
full sum, which saves a broadcast when all lanes need the result (softmax,
normalization).

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

Two shuffle rounds and **one** barrier, instead of eight barriers. If the
function is called twice in a row (for example a sum and then a sum of
squares), add a `__syncthreads()` at its end so that the second call does
not overwrite `warp_sums` while warp 0 is still reading it.

## 4. The Classic Evolution

Mark Harris' *Optimizing Parallel Reduction in CUDA* walks through seven
versions. The lessons still hold:

| # | Change | What it fixes |
|---|---|---|
| 1 | Interleaved addressing, `if (tid % (2*s) == 0)` | (baseline) heavy divergence: half the lanes of every warp idle from the first step |
| 2 | Strided index `index = 2*s*tid` | Divergence, but introduces shared-memory bank conflicts |
| 3 | Sequential addressing (above) | Bank conflicts |
| 4 | First add during the global load | Half the threads idled in the first level |
| 5 | Unroll the last warp | Barriers and loop overhead when only one warp is left (today: shuffles) |
| 6 | Complete unrolling with templates | Loop overhead |
| 7 | Many elements per thread (grid-stride) | The kernel becomes bandwidth-bound: the target |

The reduction reads each input once, so its bound is

$$
T_{\min} = \frac{4n}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $4n$ | Bytes read (float inputs) |
| $\beta$ | DRAM bandwidth |

Version 7 plus `float4` loads reaches 85–95 % of it.

## 5. Finishing Across Blocks

Blocks cannot wait for each other, so the per-block results need a second
step:

![The full reduction: per-thread sums, a reduction per block, and a final step across blocks](figures/ch03-two-level.svg)

| Method | How | Deterministic? |
|---|---|---|
| Atomics | Thread 0 of each block does `atomicAdd(out, block_sum)` | No: the order of additions varies from run to run |
| Two kernels | Kernel 1 writes $G$ partials; kernel 2 (one block) reduces them | Yes |
| Last-block | Each block writes its partial, `__threadfence()`, increments a counter; the block that sees the count reach $G$ reduces the partials | Yes |
| Cooperative groups | `grid.sync()` inside one cooperative launch | Yes |

The two-kernel version is used by most problem pages here (for example
[Tensara – Frobenius Norm](../tensara/frobenius-norm/) and
[Tensara – MSE Loss](../tensara/mse-loss/)): with $G \le 1024$ blocks the
second kernel is tiny.

## 6. Accuracy

Floating-point addition is not associative, so the GPU result differs from
a sequential CPU sum. Classical error bounds (Higham) for $n$ terms in a
format with unit roundoff $u$:

$$
\lvert \hat{s} - s \rvert \le \gamma_{n-1} \sum_i \lvert x_i \rvert \ \ (\text{sequential}), \qquad
\lvert \hat{s} - s \rvert \le \gamma_{\lceil \log_2 n \rceil} \sum_i \lvert x_i \rvert \ \ (\text{pairwise tree}), \qquad
\gamma_k = \frac{k u}{1 - k u}
$$

| Symbol | Meaning |
|---|---|
| $s, \hat{s}$ | Exact and computed sums |
| $u$ | Unit roundoff: $2^{-24}$ for fp32, $2^{-53}$ for fp64 |
| $\gamma_k$ | Error growth factor after $k$ dependent additions |

The GPU scheme (per-thread sequential chains, then a tree) sits between the
two: the chain length is about $n/p$. Practical consequences:

- with $n = 10^8$ fp32 values, a single sequential accumulator can lose
  4–5 significant digits; the tree loses almost nothing;
- accumulating the per-thread and per-block partials in `double` makes the
  result essentially exact for any size seen here;
- Kahan (compensated) summation keeps a running error term $c$:
  `y = x - c; t = sum + y; c = (t - sum) - y; sum = t;`. It costs 4 flops
  per element, which is free in a memory-bound kernel. Beware of
  `--use_fast_math`, which may reassociate it away.

## 7. Reductions in General: Monoids

Any **associative** operator $\oplus$ with an identity $e$ can be reduced
with exactly the same code:

$$
x_0 \oplus x_1 \oplus \cdots \oplus x_{n-1}, \qquad (a \oplus b) \oplus c = a \oplus (b \oplus c), \qquad e \oplus a = a
$$

| Symbol | Meaning |
|---|---|
| $\oplus$ | The combine operator |
| $e$ | Its identity element (what padding lanes contribute) |

| Reduction | State | Identity $e$ | Combine $(a \oplus b)$ |
|---|---|---|---|
| Sum | $s$ | 0 | $s_a + s_b$ |
| Max | $m$ | $-\infty$ | $\max(m_a, m_b)$ |
| arg-max (first index) | $(v, j)$ | $(-\infty, \infty)$ | The larger $v$; on ties the smaller $j$ |
| Softmax normaliser | $(m, z)$ | $(-\infty, 0)$ | $M = \max(m_a, m_b)$, $z = z_ae^{m_a - M} + z_be^{m_b - M}$ |
| Mean and variance (Welford / Chan) | $(n, \mu, M_2)$ | $(0, 0, 0)$ | See below |

Chan's parallel merge for the mean and the sum of squared deviations:

$$
n = n_a + n_b, \qquad \delta = \mu_b - \mu_a, \qquad
\mu = \mu_a + \delta\,\frac{n_b}{n}, \qquad
M_2 = M_{2,a} + M_{2,b} + \delta^2\,\frac{n_a n_b}{n}
$$

| Symbol | Meaning |
|---|---|
| $n_a, n_b$ | Element counts of the two partial results |
| $\mu_a, \mu_b$ | Their means |
| $M_{2,a}, M_{2,b}$ | Their sums of squared deviations from their own means |
| $\delta$ | Difference of the means |
| $\mu, M_2$ | Merged mean and sum of squared deviations; the variance is $M_2 / n$ |

This gives mean and variance in a single pass without the cancellation of
$\mathbb{E}[x^2] - \mathbb{E}[x]^2$. The Tensara pages implement exactly
this "accumulator struct" idea ([Argmax](../tensara/argmax/),
[Sum Dim](../tensara/sum-dim/)); the softmax monoid is the heart of
FlashAttention ([Tensara – Scaled Dot-Product Attention](../tensara/scaled-dot-attention/)).

## Practice

- [LeetGPU – Reduction](../leetgpu/004-reduction/)
- [LeetGPU – Dot Product](../leetgpu/017-dot-product/)
- [LeetGPU – Softmax](../leetgpu/005-softmax/)
- [Tensara – Argmax](../tensara/argmax/), [Tensara – Layer Norm](../tensara/layer-norm/)

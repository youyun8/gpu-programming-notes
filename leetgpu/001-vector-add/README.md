---
title: Vector Addition
platform: LeetGPU
upstream: easy/1_vector_add
url: https://leetgpu.com/challenges/vector-addition
difficulty: easy
tags: [elementwise, memory-bound, coalescing]
status: solved
---

# Vector Addition

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/vector-addition)

## Problem

Add two float32 vectors element by element. `A`, `B` and `C` are device
pointers to arrays of length $N$ ($1 \le N \le 10^8$; performance is measured
at $N = 2.5\times10^7$). The result must be written to `C`. This is the
"hello world" of GPU programming, and it is worth doing carefully: it is the
purest example of a **bandwidth-bound** kernel, and the thread-indexing
pattern here is reused by every elementwise problem on this site.

## Formulation

$$
C_i = A_i + B_i, \qquad i = 0, 1, \dots, N-1
$$

| Symbol | Meaning |
|---|---|
| $N$ | number of elements in each vector |
| $i$ | element index (0-based) |
| $A_i,\ B_i$ | $i$-th input elements, IEEE-754 float32 |
| $C_i$ | $i$-th output element, float32 |

Each output depends on exactly one element of each input, so the problem is
**embarrassingly parallel**: no two outputs share any work.

The thread that produces $C_i$ is found with the standard 1-D global index:

$$
i = b \cdot T + t, \qquad G = \left\lceil \frac{N}{T} \right\rceil
$$

| Symbol | Meaning |
|---|---|
| $t$ | thread index inside its block (`threadIdx.x`), $0 \le t < T$ |
| $b$ | block index inside the grid (`blockIdx.x`), $0 \le b < G$ |
| $T$ | threads per block (`blockDim.x`), here $T = 256$ |
| $G$ | number of blocks launched (`gridDim.x`) |
| $\lceil\cdot\rceil$ | ceiling; the last block may be partially idle, hence the guard $i < N$ |

## Approach

### Parallel decomposition

One thread per element. The launch is `vectorAdd<<<G, 256>>>` with
$G = \lceil N/256 \rceil$. Each thread computes its $i$, returns if
$i \ge N$, and otherwise does one load from `A`, one from `B`, one add and one
store to `C`.

### Why the memory access pattern matters

A warp is 32 consecutive threads, so it touches the 32 consecutive floats
$A_{32w}, \dots, A_{32w+31}$, i.e. 128 contiguous bytes. The memory system
serves this as a single fully used transaction (4 sectors of 32 bytes). This
is called **coalescing**, and it is the only optimisation that really matters
here: every byte fetched is a byte used.

### Why not do more per thread?

Grid-stride loops or `float4` vectorised loads (each thread handles 4
elements) reduce the instruction count and can help a few percent on very
large $N$. They do not change the byte count, which is what bounds the
runtime. The simple one-thread-per-element version already saturates DRAM
when $N$ is large, so it is kept for clarity.

## Cost analysis

$$
W = N, \qquad Q = 3 \cdot 4N = 12N \ \text{bytes}, \qquad I = \frac{W}{Q} = \frac{1}{12}\ \text{FLOP/byte},
\qquad T_{\min} = \frac{Q}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $W$ | work: floating-point operations (one add per element) |
| $Q$ | DRAM traffic in bytes: read $A$ and $B$, write $C$, 4 bytes each |
| $I$ | arithmetic intensity, FLOPs per byte of DRAM traffic |
| $\beta$ | sustainable DRAM bandwidth of the GPU (bytes/s) |
| $T_{\min}$ | lower bound on kernel time imposed by memory traffic |

A modern GPU needs $I$ of roughly 10–100 FLOP/byte before arithmetic becomes
the limit, so at $I = 1/12$ this kernel is **deeply memory-bound**. At the
benchmark size $N = 2.5\times10^7$, $Q = 300$ MB. On a GPU with
$\beta \approx 2\ \text{TB/s}$ that gives $T_{\min} \approx 150\ \mu s$. The right
metric for this kernel is therefore achieved GB/s, not GFLOP/s.

## Pitfalls

- **Missing bounds check.** When $N$ is not a multiple of $T$, the last
  block has threads with $i \ge N$ that must not read or write.
- **Index overflow.** `blockIdx.x * blockDim.x` is computed in 32-bit `int`.
  It is safe up to $N < 2^{31}$ (true here); beyond that, use `size_t`.
- **Timing.** The launch is asynchronous. `cudaDeviceSynchronize()` after the
  launch makes the harness observe completion (and surfaces launch errors).

## Verification

The solution is run against the platform's own reference (`torch.add`) with
`atol = rtol = 1e-5` on every LeetGPU test case, including sizes that are not
multiples of 256, using the [cuemu](../../tools/cuemu/README.md) CPU emulator
with guard pages after every buffer.

## Related

- [Matrix Addition](../008-matrix-addition/), [ReLU](../021-relu/),
  [Color Inversion](../007-color-inversion/): the same pattern in 2-D or with
  other per-element functions.
- [Tutorial 01 – Execution model](../../tutorials/01-execution-model.md) and
  [02 – Memory hierarchy](../../tutorials/02-memory-hierarchy.md).

---
title: Reduction
platform: LeetGPU
upstream: medium/4_reduction
url: https://leetgpu.com/challenges/reduction
difficulty: medium
tags: [reduction, warp-shuffle, two-pass, deterministic]
status: solved
---

# Reduction

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/reduction)

## Problem

Sum an array of $N$ float32 values into a single float
($1 \le N \le 10^8$, $\lvert x_i \rvert \le 1000$; benchmark $N = 4\,194\,304$). The
reference sums in **float64** and rounds once, with `atol = rtol = 1e-5`.
Reduction is the canonical "many inputs → one output" pattern. Softmax, norms,
losses, dot products and every `*-dim` problem on this site reuse its
building blocks.

## Formulation

$$
S = \sum_{i=0}^{N-1} x_i
$$

| Symbol | Meaning |
|---|---|
| $N$ | number of input elements |
| $x_i$ | $i$-th input value (float32) |
| $S$ | the sum, written to `output[0]` as float32 |

Addition is associative in real arithmetic, so the sum may be evaluated as any
**tree**. A parallel reduction regroups it into a hierarchy of partial sums:

$$
S = \sum_{b=0}^{B-1} \underbrace{\sum_{w=0}^{W-1} \underbrace{\sum_{\ell=0}^{31} \underbrace{\sum_{i \in \mathcal{I}(b,w,\ell)} x_i}_{\text{thread}}}_{\text{warp}}}_{\text{block } b}
$$

| Symbol | Meaning |
|---|---|
| $B$ | number of blocks in pass 1 ($\le 1024$) |
| $W$ | warps per block ($256/32 = 8$) |
| $\ell$ | lane index within a warp, $0..31$ |
| $\mathcal{I}(b,w,\ell)$ | indices visited by that thread's grid-stride loop: $i \equiv g \pmod{P}$ with $g$ the global thread id and $P = 256B$ the total thread count |

In floating point, addition is **not** associative, so different trees give
slightly different answers. The error of a sum of $N$ terms in precision $u$
is bounded by

$$
\lvert \hat S - S \rvert \le \gamma_{h}\sum_i \lvert x_i \rvert, \qquad \gamma_h = \frac{h\,u}{1 - h\,u}
$$

| Symbol | Meaning |
|---|---|
| $\hat S$ | computed sum |
| $u$ | unit roundoff: $2^{-24}$ for float32, $2^{-53}$ for float64 |
| $h$ | height of the summation tree: $N-1$ for a sequential loop, $\approx\log_2 N$ for a balanced tree |
| $\gamma_h$ | standard error-growth constant (Higham) |

A tree is therefore not only faster but also *more accurate* than a
sequential loop. Doing the upper levels in float64 makes their contribution
to the error negligible.

## Approach

### Pass 1: `partialSums` (≤ 1024 Blocks × 256 Threads)

1. **Grid-stride loop with `float4` loads.** Thread $g$ reads vectors
   $g, g+P, g+2P, \dots$ of 4 floats (16 bytes per load, fully coalesced) and
   accumulates in a float32 register. A scalar tail loop handles the last
   $N \bmod 4$ elements.
2. **Warp level.** Five `__shfl_down_sync` steps with offsets 16, 8, 4, 2, 1
   fold 32 lane values into lane 0, in float64. Shuffles exchange registers
   directly, with no shared memory and no barriers.
3. **Block level.** Lane 0 of each warp writes its value to `warp_sums[w]`.
   After one `__syncthreads()`, warp 0 reduces the 8 values with the same
   shuffles.
4. Thread 0 writes the block's float64 partial to the `__device__` array
   `g_partials[b]`.

### Pass 2: `finalSum` (1 Block)

It adds the $B$ partials with the same block reduction in float64, then
rounds once to float32. Because the second kernel starts only after the
first finishes (same stream), no grid-wide synchronisation is needed.

### Why Two Passes and Not `atomicAdd`?

A single kernel in which every block does `atomicAdd(output, partial)` also
works. However, float atomics complete in a nondeterministic order, so the
result can change from run to run in its last bits. The two-pass design is
**deterministic**, and the second pass costs a few microseconds.

## Cost Analysis

$$
Q = 4N \ \text{bytes}, \qquad W = N - 1, \qquad I = \frac{W}{Q} \approx \frac{1}{4}, \qquad T_{\min} = \frac{4N}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes read (each input read exactly once) |
| $W$ | additions |
| $I$ | arithmetic intensity (FLOP/byte) |
| $\beta$ | DRAM bandwidth |
| $T_{\min}$ | bandwidth lower bound on the runtime |

At the benchmark size, $Q = 16.8$ MB, so $T_{\min} \approx 8\ \mu s$ at 2 TB/s.
At this size, launch overhead (~2–5 µs per kernel) is comparable to the
transfer time. That is why the grid is capped at 1024 blocks and only two
launches are used.

## Pitfalls

- **Misaligned `float4`.** `input` comes from `cudaMalloc`, which is 256-byte
  aligned, so casting to `float4*` is safe. It would not be safe for an
  arbitrary offset pointer.
- **Inactive warps in the block step.** Only the first $W$ entries of
  `warp_sums` are valid. Threads with `threadIdx.x >= W` must contribute 0.
- **Returning early before `__syncthreads()`.** Every thread must reach the
  barrier inside `blockReduceSum`, even threads that had no elements.
- **Tiny $N$.** $N < 4$ gives zero vectors; the scalar tail handles all
  elements, and the grid is clamped to at least one block.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) against the
float64 reference, including $N = 1$, sizes not divisible by 4, and
near-cancelling inputs. They also pass under `--reverse` thread scheduling,
which checks that no barrier is missing.

## Related

- [Dot Product](../017-dot-product/), [Softmax](../005-softmax/), [RMS Norm](../050-rms-normalization/).
- [Tutorial 03 – Parallel reduction](../../tutorials/03-parallel-reduction.md).

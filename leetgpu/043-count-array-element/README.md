---
title: Count Array Element
platform: LeetGPU
upstream: medium/43_count_array_element
url: https://leetgpu.com/challenges/count-array-element
difficulty: medium
tags: [reduction, counting, warp-intrinsics, atomics]
status: solved
---

# Count Array Element

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/count-array-element)

## Problem

Count how many of $N$ int32 values equal $K$ ($1 \le N \le 10^8$; benchmark
$N = 10^8$). The result is an exact int32. It is a reduction whose map step
is a comparison, and a showcase for the **single-instruction warp
reduction** `__reduce_add_sync` (sm_80+).

## Formulation

$$
\text{count} = \sum_{i=0}^{N-1} \bigl[\,x_i = K\,\bigr]
$$

| Symbol | Meaning |
|---|---|
| $N$ | number of elements |
| $x_i$ | input values (int32) |
| $K$ | the value to count |
| $[\cdot]$ | Iverson bracket (1 if true, else 0) |
| count | result, written to `output[0]` |

Integer addition is associative **and exact**, so any reduction order gives
the same answer. Atomics are therefore safe here, unlike for float sums.

## Approach

1. `cudaMemset(output, 0)`.
2. `countEqual` (≤ 2048 blocks × 256 threads):
   - A grid-stride loop over `int4` vectors adds the four comparisons
     `(v.x == K) + … `. `bool` converts to 0/1 without a branch. A scalar
     loop handles the tail.
   - `__reduce_add_sync(0xffffffff, count)` sums the 32 lane counts in **one
     instruction** (`REDUX.SUM`), instead of 5 shuffle+add steps.
   - Lane 0 of each warp does a single `atomicAdd(output, count)` if the
     count is non-zero.

At most $2048 \cdot 8 = 16\,384$ global atomics hit one address. That is
negligible next to reading 400 MB.

## Cost Analysis

$$
Q = 4N \ \text{bytes}, \qquad T_{\min} = \frac{4N}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes (read every element once) |
| $\beta$ | DRAM bandwidth |

Benchmark: 400 MB, i.e. ≈ 200 µs at 2 TB/s.

## Pitfalls

- **Forgetting to zero `output`.** It then accumulates garbage.
- **`__reduce_add_sync` requires sm_80.** On older GPUs, use a shuffle loop.
  (The [cuemu](../../tools/cuemu/README.md) emulator implements it.)
- **The benchmark's $K$ (501 010) is outside the value range**, so the
  correct answer there is 0. The kernel handles that naturally.

## Verification

Exact equality on all LeetGPU cases in [cuemu](../../tools/cuemu/README.md).

## Related

- [Count 2D](../044-count-2d-array-element/), [Count 3D](../045-count-3d-array-element/),
  [Histogramming](../013-histogramming/), [Reduction](../004-reduction/).

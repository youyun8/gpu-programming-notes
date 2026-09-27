---
title: Subarray Sum
platform: LeetGPU
upstream: medium/47_subarray_sum
url: https://leetgpu.com/challenges/subarray-sum
difficulty: medium
tags: [reduction, integer, warp-intrinsics]
status: solved
---

# Subarray Sum

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/subarray-sum)

## Problem

Sum the int32 values `input[S..E]` (inclusive) of an array of length $N$
($N \le 10^8$, values in $[1, 10]$; benchmark $N = 10^8$). The result is an
exact int32. Integer addition is exact and associative, which allows a
simpler reduction design than for floats.

## Visual Overview

![Subarray sum: add the values x[S] … x[E] with an exact integer reduction](figure.svg)

Only the highlighted range is read. Integer addition is exact, so the threads
may combine their partial sums in any order, including with atomics.

## Formulation

$$
\text{out} = \sum_{i=S}^{E} x_i
$$

| Symbol | Meaning |
|---|---|
| $N$ | Array length |
| $x_i$ | Input values (int32) |
| $S,\ E$ | Inclusive 0-based start and end indices, $0 \le S \le E < N$ |
| out | Exact sum, written to `output[0]` |

The maximum possible value is $10 \cdot 10^8 = 10^9 < 2^{31}$, so int32 cannot
overflow.

## Approach

- `cudaMemset(output, 0)`.
- Launch $\min(\lceil (E-S+1)/256\rceil, 2048)$ blocks. Thread $g$ sums
  $x_{S+g}, x_{S+g+P}, \dots$ (grid-stride, where $P$ is the total number of
  threads); consecutive threads read consecutive addresses.
- `__reduce_add_sync` folds the warp in one instruction, and lane 0 issues
  one `atomicAdd`.

Since integer addition is exact, the nondeterministic order of the atomics
does not matter. No second pass is needed, unlike the float reductions.

## Cost Analysis

$$
Q = 4(E - S + 1)\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes (only the requested range is read) |
| $\beta$ | DRAM bandwidth |

## Pitfalls

- **Unaligned start.** `input + S` is not 16-byte aligned in general, so
  `int4` loads would need a scalar prologue. The scalar grid-stride loop is
  simpler and still coalesced.
- **Empty grid** is impossible, since $E \ge S$ guarantees at least one element.

## Verification

Exact equality on all LeetGPU cases in [cuemu](../../tools/cuemu/README.md),
including $S = E$.

## Related

- [2D Subarray Sum](../048-2d-subarray-sum/), [3D Subarray Sum](../049-3d-subarray-sum/),
  [Count Array Element](../043-count-array-element/).

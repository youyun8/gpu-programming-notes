---
title: Reverse Array
platform: LeetGPU
upstream: easy/19_reverse_array
url: https://leetgpu.com/challenges/reverse-array
difficulty: easy
tags: [in-place, memory-bound, race-conditions]
status: solved
---

# Reverse Array

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/reverse-array)

## Problem

Reverse a float32 array of length $N$ **in place** ($1 \le N \le 10^8$;
benchmark $N = 2.5\times10^7$). The interesting part is doing it without a
second buffer and without a data race.

## Formulation

$$
x'_i = x_{N-1-i}, \qquad 0 \le i < N
$$

| Symbol | Meaning |
|---|---|
| $N$ | array length |
| $x_i$ | value at index $i$ before the call |
| $x'_i$ | value at index $i$ after the call |

The map $i \mapsto N-1-i$ is an **involution**: it pairs index $i$ with its
mirror $j = N-1-i$, and applying it twice is the identity. The update
therefore decomposes into $\lfloor N/2\rfloor$ independent swaps:

$$
(x_i,\ x_j) \leftarrow (x_j,\ x_i), \qquad j = N-1-i,\ \ 0 \le i < \lfloor N/2 \rfloor
$$

| Symbol | Meaning |
|---|---|
| $j$ | mirror index of $i$ |
| $\lfloor N/2 \rfloor$ | number of swaps; for odd $N$ the middle element $x_{(N-1)/2}$ is its own mirror and stays put |

## Approach

Launch $\lceil \lfloor N/2\rfloor / 256\rceil$ blocks. Thread $i < \lfloor N/2\rfloor$
loads both $x_i$ and $x_j$ into registers, then stores them swapped. Each
memory location is read and written by exactly one thread, so there is **no
race**.

### Coalescing

A warp's 32 threads read $x_i, \dots, x_{i+31}$ (ascending, contiguous) and
$x_{j-31}, \dots, x_j$ (descending, but still the same 128-byte segment).
The hardware coalesces by address set, not by order, so both halves are
fully coalesced.

## Cost analysis

$$
Q = 2 \cdot 4N \ \text{bytes}, \qquad T_{\min} = \frac{8N}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM traffic: every element is read once and written once |
| $\beta$ | DRAM bandwidth |

Benchmark: $Q = 200$ MB, so $T_{\min} \approx 100\ \mu s$ at 2 TB/s.

## Pitfalls

- **Racing copy.** Launching $N$ threads that each do `x[N-1-i] = x[i]`
  overwrites values that other threads have not read yet. The result depends
  on scheduling.
- **Zero-size grid.** For $N = 1$ there is nothing to swap. The launch is
  skipped, because a grid of 0 blocks is an invalid configuration.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md), including
$N = 1$, $N = 2$ and odd lengths. `--reverse` scheduling confirms that the
result does not depend on thread order.

## Related

- [Matrix Transpose](../003-matrix-transpose/) (another pure data movement),
  [Interleave](../063-interleave/).

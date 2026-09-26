---
title: Interleave Arrays
platform: LeetGPU
upstream: easy/63_interleave
url: https://leetgpu.com/challenges/interleave-arrays
difficulty: easy
tags: [memory-bound, vectorized, data-movement]
status: solved
---

# Interleave Arrays

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/interleave-arrays)

## Problem

Interleave two float32 arrays of length $N$ into one of length $2N$:
$[a_0, b_0, a_1, b_1, \dots]$ ($N \le 5\times10^7$; benchmark
$N = 2.5\times10^7$). This is a pure data-layout transformation: converting
structure-of-arrays (SoA) to array-of-structures (AoS), as needed for
complex numbers, (x, y) points, or the `float2` layout expected by many APIs.

## Formulation

$$
o_{2i} = a_i, \qquad o_{2i+1} = b_i, \qquad 0 \le i < N
$$

| Symbol | Meaning |
|---|---|
| $N$ | length of each input |
| $a_i,\ b_i$ | inputs `A[i]`, `B[i]` (float32) |
| $o_k$ | output, $0 \le k < 2N$ |

## Approach

Thread $i$ reads $a_i$ and $b_i$ and writes them together as **one
`float2`** at `output[2i]`:

- the reads of $a$ and $b$ are contiguous across the warp (two 128-byte
  segments);
- the write is 32 consecutive `float2`s = 256 contiguous bytes, one fully
  coalesced 8-byte store per thread.

With two separate 4-byte stores to `output[2i]` and `output[2i+1]`, each
store instruction would touch 256 bytes but use only half of them. That
wastes half of the store bandwidth per instruction (L2 merges the halves
eventually, but the request count doubles).

## Cost Analysis

$$
Q = 4N + 4N + 8N = 16N\ \text{bytes}, \qquad T_{\min} = \frac{16N}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes: read $a$ and $b$, write $2N$ floats |
| $\beta$ | DRAM bandwidth |

Benchmark: 400 MB, i.e. ≈ 200 µs at 2 TB/s.

## Pitfalls

- **Alignment.** `output` reinterpreted as `float2*` needs 8-byte alignment,
  which `cudaMalloc` provides.
- **Index width.** $2N \le 10^8$ fits in `int`.

## Verification

Exact equality on all LeetGPU cases in [cuemu](../../tools/cuemu/README.md).

## Related

- [Matrix Transpose](../003-matrix-transpose/) (the 2-D generalisation of layout changes), [Matrix Copy](../031-matrix-copy/).

---
title: 2D Subarray Sum
platform: LeetGPU
upstream: medium/48_2d_subarray_sum
url: https://leetgpu.com/challenges/2d-subarray-sum
difficulty: medium
tags: [reduction, integer, warp-intrinsics]
status: solved
---

# 2D Subarray Sum

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/2d-subarray-sum)

## Problem

Sum the rectangle `input[S_ROW..E_ROW][S_COL..E_COL]` (inclusive) of an
$N \times M$ int32 matrix ($N, M \le 10^4$, values in $[1, 10]$; benchmark
$N = M = 10^4$). The result is an exact int32.

## Formulation

$$
\text{out} = \sum_{r = r_0}^{r_1}\ \sum_{c = c_0}^{c_1} x_{r c}, \qquad
\text{flattened: } i \mapsto (r, c) = \Bigl(r_0 + \bigl\lfloor i / w \bigr\rfloor,\ c_0 + (i \bmod w)\Bigr),\ 0 \le i < h w
$$

| Symbol | Meaning |
|---|---|
| $N,\ M$ | matrix rows and columns |
| $x_{rc}$ | element at offset $rM + c$ |
| $r_0, r_1$ | `S_ROW`, `E_ROW` |
| $c_0, c_1$ | `S_COL`, `E_COL` |
| $h,\ w$ | rectangle height $r_1 - r_0 + 1$ and width $c_1 - c_0 + 1$ |
| $i$ | flat index over the rectangle, column-fastest |

## Approach

The rectangle is flattened to a 1-D index space of $h w$ elements with the
column index fastest. Consecutive threads then read consecutive elements of
a row, which is coalesced, apart from one row break per $w$ elements.
Otherwise it is the kernel from [Subarray Sum](../047-subarray-sum/):
grid-stride accumulation, `__reduce_add_sync`, and one `atomicAdd` per warp.
The flat index is 64-bit ($h w \le 10^8$ fits in 32 bits, but the product is
formed safely).

## Cost Analysis

$$
Q = 4hw\ \text{bytes (useful)}, \qquad \text{sectors touched} \approx h\left\lceil\frac{4w}{32}\right\rceil + h
$$

| Symbol | Meaning |
|---|---|
| $Q$ | bytes actually needed |
| sectors | 32-byte DRAM sectors fetched; each row segment may straddle one extra sector at each end |

For narrow rectangles (small $w$), the per-row overhead dominates. For the
full-width benchmark, efficiency is essentially 100%.

## Pitfalls

- **2-D thread mapping** (one thread per row looping over columns) makes
  consecutive threads read addresses $M$ apart, which is uncoalesced.
- **Integer division per element** (`i / w`, `i % w`) is not free, but it is
  hidden behind the memory latency.

## Verification

Exact equality on all LeetGPU cases in [cuemu](../../tools/cuemu/README.md),
including 1 × 1 and full-matrix rectangles.

## Related

- [Subarray Sum](../047-subarray-sum/), [3D Subarray Sum](../049-3d-subarray-sum/).

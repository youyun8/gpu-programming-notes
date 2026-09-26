---
title: Count 2D Array Element
platform: LeetGPU
upstream: medium/44_count_2d_array_element
url: https://leetgpu.com/challenges/count-2d-array-element
difficulty: medium
tags: [reduction, counting, warp-intrinsics]
status: solved
---

# Count 2D Array Element

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/count-2d-array-element)

## Problem

Count occurrences of $K$ in an $N \times M$ int32 matrix
($1 \le N, M \le 10^4$, values in $[1, 100]$; benchmark
$N = M = 10^4$, $K = 1$). The matrix is contiguous, so the 2-D shape only
determines the element count.

## Formulation

$$
\text{count} = \sum_{r=0}^{N-1}\sum_{c=0}^{M-1} \bigl[\,x_{rc} = K\,\bigr] = \sum_{i=0}^{NM-1} \bigl[\,x_i = K\,\bigr], \qquad i = rM + c
$$

| Symbol | Meaning |
|---|---|
| $N,\ M$ | Rows and columns |
| $x_{rc}$ | Element at row $r$, column $c$ (int32) |
| $i$ | Flattened row-major index |
| $K$ | Value to count |
| Count | Exact result in `output[0]` |

## Approach

This is the flat counting kernel of [Count Array Element](../043-count-array-element/)
(`int4` loads, `__reduce_add_sync`, one atomic per warp), with the element
count computed as a **64-bit** `long long`. $N M$ is at most $10^8$ here,
but writing it in 64-bit keeps the kernel safe for larger shapes. Loop
indices are 64-bit as well.

## Cost Analysis

$$
Q = 4NM \ \text{bytes}, \qquad T_{\min} = \frac{4NM}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes |
| $\beta$ | DRAM bandwidth |

Benchmark: 400 MB, i.e. ≈ 200 µs at 2 TB/s. With values uniform in $[1, 100]$,
about 1% of elements match. The per-warp atomics are sparse and free.

## Pitfalls

- **Launching a 2-D grid** and indexing `x[r][c]` works, but adds index
  arithmetic and complicates vectorisation.
- **`int` overflow of $N \cdot M$** for larger variants: compute it in 64-bit.

## Verification

Exact equality on all LeetGPU cases in [cuemu](../../tools/cuemu/README.md).

## Related

- [Count Array Element](../043-count-array-element/), [Count 3D](../045-count-3d-array-element/).

---
title: Count 3D Array Element
platform: LeetGPU
upstream: medium/45_count_3d_array_element
url: https://leetgpu.com/challenges/count-3d-array-element
difficulty: medium
tags: [reduction, counting, warp-intrinsics]
status: solved
---

# Count 3D Array Element

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/count-3d-array-element)

## Problem

Count occurrences of $P$ in an $N \times M \times K$ int32 tensor
($1 \le N, M, K \le 1000$; benchmark $500^3$). As in the 2-D version, the
tensor is contiguous and the problem reduces to a flat count over
$NMK \le 10^9$ elements. That count can exceed $2^{31}$ **only as a byte
count**, which is why the offsets are 64-bit.

## Formulation

$$
\text{count} = \sum_{a=0}^{N-1}\sum_{b=0}^{M-1}\sum_{c=0}^{K-1}\bigl[\,x_{abc} = P\,\bigr], \qquad \text{offset}(a, b, c) = (aM + b)K + c
$$

| Symbol | Meaning |
|---|---|
| $N,\ M,\ K$ | tensor dimensions |
| $x_{abc}$ | element (int32) |
| $P$ | value to count |
| offset | flat row-major index |
| count | exact result in `output[0]` |

## Approach

The same kernel as [Count 2D](../044-count-2d-array-element/): a 64-bit
element count, grid-stride `int4` loads, `__reduce_add_sync`, and one
`atomicAdd` per warp.

## Cost Analysis

$$
Q = 4NMK \ \text{bytes}, \qquad T_{\min} = \frac{4NMK}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes |
| $\beta$ | DRAM bandwidth |

Benchmark: 500 MB, i.e. ≈ 250 µs at 2 TB/s. At the maximum size
($10^9$ elements = 4 GB), the 64-bit indexing is necessary.

## Pitfalls

- **32-bit loop counters** overflow at $NMK > 2^{31}/4$ vectors, i.e. when
  indexing bytes or large strides.
- **Zeroing the output** before accumulating.

## Verification

Exact equality on all LeetGPU cases in [cuemu](../../tools/cuemu/README.md).

## Related

- [Count Array Element](../043-count-array-element/), [Count 2D](../044-count-2d-array-element/).

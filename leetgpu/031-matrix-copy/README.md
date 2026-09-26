---
title: Matrix Copy
platform: LeetGPU
upstream: easy/31_matrix_copy
url: https://leetgpu.com/challenges/matrix-copy
difficulty: easy
tags: [memory-bound, vectorized, bandwidth]
status: solved
---

# Matrix Copy

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/matrix-copy)

## Problem

Copy an $N \times N$ float32 matrix `A` into `B` ($1 \le N \le 4096$;
benchmark $N = 4096$). There is nothing to compute. The problem measures how
close a kernel gets to the **peak copy bandwidth** of the GPU, which is the
ceiling for every memory-bound problem on this site.

## Formulation

$$
B_k = A_k, \qquad 0 \le k < N^2
$$

| Symbol | Meaning |
|---|---|
| $N$ | matrix side length |
| $k$ | flattened row-major index |
| $A_k,\ B_k$ | source and destination elements (float32) |

## Approach

- Flat 1-D view of the contiguous matrices.
- Thread $t < \lfloor N^2/4\rfloor$ copies one `float4` (16-byte load +
  16-byte store); threads $t < N^2 \bmod 4$ copy the scalar tail.
- One element group per thread is enough at this size: $4096^2/4 = 4.2$M
  threads keep every SM saturated with memory requests.

`cudaMemcpy(…, cudaMemcpyDeviceToDevice)` would do the same using the copy
engine or an internal kernel, but the exercise is the kernel.

### What Limits a Copy

Achieved bandwidth depends on the number of **bytes in flight**. By
Little's law:

$$
\text{bytes in flight} = \beta \times \lambda
$$

| Symbol | Meaning |
|---|---|
| $\beta$ | target bandwidth (bytes/s) |
| $\lambda$ | DRAM latency (roughly 500–800 ns) |

At $\beta = 2$ TB/s and $\lambda \approx 600$ ns, about 1.2 MB must be
outstanding at all times. `float4` accesses quadruple the bytes per
instruction, which makes it easy to reach that with ordinary occupancy.

## Cost Analysis

$$
Q = 2 \cdot 4N^2 \ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | bytes moved: read $N^2$ floats, write $N^2$ floats |
| $T_{\min}$ | bandwidth lower bound |

$N = 4096$: $Q = 134$ MB, so $T_{\min} \approx 67\ \mu s$ at 2 TB/s. Compare
your transpose ([Matrix Transpose](../003-matrix-transpose/)) against this
number.

## Pitfalls

- **Aliasing.** Source and destination never overlap here. A general
  `memmove` would need to handle overlap.
- **Tail threads** for odd $N$ (then $N^2$ is odd).

## Verification

Exact equality on all LeetGPU cases in [cuemu](../../tools/cuemu/README.md).

## Related

- [Matrix Transpose](../003-matrix-transpose/), [Vector Addition](../001-vector-add/).
- [Tutorial 02 – Memory hierarchy](../../tutorials/02-memory-hierarchy.md).

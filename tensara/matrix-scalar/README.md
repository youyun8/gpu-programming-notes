---
title: Matrix Scalar Multiplication
platform: Tensara
upstream: matrix-scalar
url: https://tensara.org/problems/matrix-scalar
difficulty: easy
tags: [elementwise, float4]
status: solved
---

# Matrix Scalar Multiplication

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/matrix-scalar)

## Problem

Multiply an $n\times n$ float32 matrix by a scalar $s$ ($n$ = 8192 or
9216, $s \in \{0.1, 0.2, -0.3, 0.4, -0.5\}$). The check is `rtol = 1e-4`,
`atol = 7e-6`.

## Formulation

$$
C_{ij} = s\,A_{ij}, \qquad 0 \le i, j < n
$$

| Symbol | Meaning |
|---|---|
| $A$ | input matrix, $n\times n$ float32 |
| $s$ | scalar multiplier |
| $C$ | output matrix |

## Approach

All Tensara elementwise problems share one kernel shape:

1. **`float4` grid-stride loop.** The buffer is viewed as $\lfloor n/4 \rfloor$
   16-byte vectors; each iteration loads one `float4`, applies the scalar
   function to the four lanes, and stores one `float4`. `cudaMalloc`
   returns 256-byte-aligned pointers, so the reinterpretation is safe.
2. **Scalar tail** for the last $n \bmod 4$ elements.
3. **Launch** 256-thread blocks, capped at 4096 blocks; the grid-stride
   loop covers any size, and 4096 × 256 threads are enough to saturate DRAM.
4. The function is a `__forceinline__` device function, so the loop body is
   branch-free apart from the select in the function itself.

One `FMUL` per element; the result is exactly the correctly rounded product, identical to PyTorch's.

## Cost Analysis

$$
n = n^2, \qquad Q = 8\,n\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $n$ | number of elements |
| $Q$ | compulsory DRAM traffic: read the input(s) once, write the output once |
| $\beta$ | DRAM bandwidth (about 2–3 TB/s on current data-centre GPUs) |
| $T_{\min}$ | bandwidth lower bound on the kernel time |

For $n = 9216$: $Q = 680$ MB, about 0.34 ms at 2 TB/s.

## Pitfalls

- **Signature**: the matrix is square, so only one size `n` is passed; the
  element count is $n^2$ (compute it in `size_t`: $9216^2 = 85$ M fits, but
  larger sizes overflow `int` indexing of bytes).
- The scalar is passed by value, not as a device pointer.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Diagonal Matmul](../diagonal-matmul/), [Vector Addition](../vector-addition/).

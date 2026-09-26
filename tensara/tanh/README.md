---
title: Tanh
platform: Tensara
upstream: tanh
url: https://tensara.org/problems/tanh
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# Tanh

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/tanh)

## Problem

Apply the hyperbolic tangent elementwise to an $M\times N$ float32
matrix, matching `torch.tanh`. Test matrices go from $4096\times4096$ to $8192\times8192$ (up to 67 M elements). The check is `rtol = 1e-4`,
`atol = 6e-5`.

## Formulation

$$
C_{ij} = \tanh(A_{ij}), \qquad \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} = 2\sigma(2x) - 1
$$

| Symbol | Meaning |
|---|---|
| $A$ | input matrix, $M\times N$ float32, row-major |
| $C$ | output matrix, same shape |
| $x$ | one input element $A_{ij}$ |
| $\tanh$ | hyperbolic tangent, range $(-1, 1)$ |
| $\sigma$ | logistic sigmoid |

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

The code calls CUDA's `tanhf`, which handles small $|x|$ with a polynomial (avoiding the cancellation of $e^x - e^{-x}$) and saturates to $\pm1$ for large $|x|$. The hardware `tanh.approx.f32` (sm_75+) is faster but has about $2^{-11}$ relative error, too coarse for `rtol = 1e-4` near 0.

## Cost Analysis

$$
n = MN, \qquad Q = 8\,n\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $n$ | number of elements |
| $Q$ | compulsory DRAM traffic: read the input(s) once, write the output once |
| $\beta$ | DRAM bandwidth (about 2–3 TB/s on current data-centre GPUs) |
| $T_{\min}$ | bandwidth lower bound on the kernel time |

For $8192\times8192$: $Q = 537$ MB, about 0.27 ms at 2 TB/s.

## Pitfalls

- **Naive formula**: $(e^x - e^{-x})/(e^x + e^{-x})$ is $\infty/\infty =$ NaN
  for $|x| > 88$ and inaccurate near 0.
- **Fast math** replaces `tanhf` with the approximate instruction.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Sigmoid](../sigmoid/), [GELU](../gelu/), [Hard Sigmoid](../hard-sigmoid/).

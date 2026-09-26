---
title: Leaky ReLU
platform: Tensara
upstream: leaky-relu
url: https://tensara.org/problems/leaky-relu
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# Leaky ReLU

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/leaky-relu)

## Problem

Apply Leaky ReLU with a runtime slope $\alpha$ (0.01 … 0.2 in the
tests) elementwise to an $M\times N$ float32 matrix ($4096\times4096$ and
$6144\times4096$), matching `F.leaky_relu(x, alpha)`. The check is
`rtol = 1e-4`, `atol = 1e-6`.

## Formulation

$$
C_{ij} = \begin{cases} x, & x > 0 \\ \alpha x, & x \le 0 \end{cases} = \max(x, 0) + \alpha\,\min(x, 0), \qquad x = A_{ij}
$$

| Symbol | Meaning |
|---|---|
| $A$ | input matrix, $M\times N$ float32, row-major |
| $C$ | output matrix, same shape |
| $x$ | one input element $A_{ij}$ |
| $\alpha$ | negative-side slope, $0 < \alpha < 1$ |

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

The select `x > 0.0f ? x : alpha * x` compiles to a multiply and a predicated move; there is no divergence.

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

For $6144\times4096$: $Q = 201$ MB, about 0.1 ms at 2 TB/s.

## Pitfalls

- **Argument order**: the signature is `(input, alpha, output, n, m)`, with
  $\alpha$ between the two pointers.
- **Tight `atol = 1e-6`**: the result must be exactly $\alpha x$ in fp32, so
  do not compute it as `x * (x > 0 ? 1 : alpha)` with a rounded constant.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [ReLU](../relu/), [ELU](../elu/), LeetGPU [Leaky ReLU](../../leetgpu/023-leaky-relu/).

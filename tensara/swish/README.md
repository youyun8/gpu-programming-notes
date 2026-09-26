---
title: Swish
platform: Tensara
upstream: swish
url: https://tensara.org/problems/swish
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# Swish

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/swish)

## Problem

Apply Swish (SiLU) elementwise to an $M\times N$ float32 matrix; the
reference is `x * torch.sigmoid(x)`. Test matrices go from $4096\times4096$ to $8192\times8192$ (up to 67 M elements). The check is `rtol = 1e-4`,
`atol = 4e-5`.

## Formulation

$$
C_{ij} = x\,\sigma(x) = \frac{x}{1 + e^{-x}}, \qquad x = A_{ij}
$$

| Symbol | Meaning |
|---|---|
| $A$ | Input matrix, $M\times N$ float32, row-major |
| $C$ | Output matrix, same shape |
| $x$ | One input element $A_{ij}$ |
| $\sigma$ | Logistic sigmoid |

Swish has a small negative dip, with minimum

$$
\min_x x\,\sigma(x) \approx -0.2785 \quad \text{at}\ x \approx -1.2785
$$

| Symbol | Meaning |
|---|---|
| $x$ | The input where Swish reaches its minimum |

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

The code writes `x / (1.0f + expf(-x))`: one exponential and one division, the same cost as sigmoid.

## Cost Analysis

$$
n = MN, \qquad Q = 8\,n\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $n$ | Number of elements |
| $Q$ | Compulsory DRAM traffic: read the input(s) once, write the output once |
| $\beta$ | DRAM bandwidth (about 2–3 TB/s on current data-centre GPUs) |
| $T_{\min}$ | Bandwidth lower bound on the kernel time |

For $8192\times8192$: $Q = 537$ MB, about 0.27 ms at 2 TB/s.

## Pitfalls

- **Large negative $x$**: $e^{-x} = \infty$ gives $x/\infty = -0$, which is
  the correct limit.
- **Reference order**: PyTorch computes $\sigma(x)$ first and multiplies;
  dividing directly differs by at most 1–2 ulp.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Sigmoid](../sigmoid/), [MatMul + Swish](../matmul-swish/), LeetGPU [SiLU](../../leetgpu/052-silu/).

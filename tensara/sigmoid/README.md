---
title: Sigmoid
platform: Tensara
upstream: sigmoid
url: https://tensara.org/problems/sigmoid
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# Sigmoid

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/sigmoid)

## Problem

Apply the logistic sigmoid elementwise to an $M\times N$ float32 matrix,
matching `torch.sigmoid`. Test matrices go from $4096\times4096$ to $8192\times8192$ (up to 67 M elements). The check is `rtol = 1e-4`, `atol = 6e-5`.

## Formulation

$$
C_{ij} = \sigma(A_{ij}), \qquad \sigma(x) = \frac{1}{1 + e^{-x}}
$$

| Symbol | Meaning |
|---|---|
| $A$ | Input matrix, $M\times N$ float32, row-major |
| $C$ | Output matrix, same shape |
| $x$ | One input element $A_{ij}$ |
| $\sigma$ | Logistic sigmoid, range $(0, 1)$ |

The formula is safe in fp32 at both ends:

$$
x \to +\infty:\ e^{-x} \to 0,\ \sigma \to 1; \qquad
x \to -\infty:\ e^{-x} \to +\infty,\ \sigma = 1/\infty = 0
$$

| Symbol | Meaning |
|---|---|
| $e^{-x}$ | Overflows to $+\infty$ for $x < -88.7$, which still gives the correct limit 0 |

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

`1.0f / (1.0f + expf(-x))`: one `expf` (a range reduction plus `ex2.approx` and a correction) and one division.

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

- **Do not "fix" overflow** with a branch: $1/(1+\infty) = 0$ is exact.
- For large negative $x$, the relative error of $\sigma$ is dominated by
  `expf`'s; with `-use_fast_math` (`__expf`) the absolute error stays tiny,
  which is what the tolerance measures.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Swish](../swish/), [Hard Sigmoid](../hard-sigmoid/), [Tanh](../tanh/), LeetGPU [Sigmoid](../../leetgpu/068-sigmoid/).

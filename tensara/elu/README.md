---
title: ELU
platform: Tensara
upstream: elu
url: https://tensara.org/problems/elu
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# ELU

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/elu)

## Problem

Apply the Exponential Linear Unit elementwise to an $M\times N$ float32
matrix, with a runtime parameter $\alpha$ (1.0 in the tests), matching
`F.elu(x, alpha)`. Test matrices go from $4096\times4096$ to $8192\times8192$ (up to 67 M elements). The check is `rtol = 1e-4`, `atol = 5e-5`.

## Formulation

$$
C_{ij} = \operatorname{ELU}_\alpha(A_{ij}), \qquad
\operatorname{ELU}_\alpha(x) = \begin{cases} x, & x > 0 \\ \alpha\,(e^{x} - 1), & x \le 0 \end{cases}
$$

| Symbol | Meaning |
|---|---|
| $A$ | Input matrix, $M\times N$ float32, row-major |
| $C$ | Output matrix, same shape |
| $x$ | One input element $A_{ij}$ |
| $\alpha$ | Saturation level: $\operatorname{ELU}_\alpha(x) \to -\alpha$ as $x \to -\infty$ |
| $e^x - 1$ | Computed with `expm1f` |

Near $x = 0^-$ the naive form $e^x - 1$ suffers cancellation: for
$x = -10^{-6}$, `expf(x)` rounds to $1 - 2^{-24}\cdot k$ and the subtraction
keeps only a few correct bits. `expm1f` evaluates

$$
\operatorname{expm1}(x) = x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots
$$

| Symbol | Meaning |
|---|---|
| $\operatorname{expm1}(x)$ | $e^x - 1$ computed without forming $e^x$ first, accurate to about 1 ulp |

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

For $8192\times8192$: $Q = 537$ MB, about 0.27 ms at 2 TB/s. `expm1f` costs a few dozen instructions per element, still far less than the ~100 instructions per element a GPU can issue in the time it takes to move 8 bytes.

## Pitfalls

- **Precision near zero**: use `expm1f`, not `expf(x) - 1.0f`.
- **Condition** is $x > 0$ for the linear branch; at $x = 0$ both branches
  give 0.
- **Parameter order** in the signature is `(input, output, n, m, alpha)`.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [SELU](../selu/), [Leaky ReLU](../leaky-relu/), [ReLU](../relu/), [GELU](../gelu/).

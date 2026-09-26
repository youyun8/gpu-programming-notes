---
title: GELU
platform: Tensara
upstream: gelu
url: https://tensara.org/problems/gelu
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# GELU

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/gelu)

## Problem

Apply GELU with the **tanh approximation** elementwise to an
$M\times N$ float32 matrix (`F.gelu(x, approximate="tanh")`). Test matrices go from $4096\times4096$ to $8192\times8192$ (up to 67 M elements). The
check is `rtol = 1e-4`, `atol = 2e-5`.

## Formulation

The exact GELU weights $x$ by the probability that a standard normal
variable is below $x$:

$$
\operatorname{GELU}(x) = x\,\Phi(x) = \frac{x}{2}\left(1 + \operatorname{erf}\frac{x}{\sqrt{2}}\right)
$$

| Symbol | Meaning |
|---|---|
| $x$ | One input element |
| $\Phi(x)$ | Standard normal cumulative distribution function |
| $\operatorname{erf}$ | Gauss error function |

This problem asks for the tanh approximation:

$$
C_{ij} = \frac{x}{2}\Bigl(1 + \tanh\bigl(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\bigr)\Bigr), \qquad x = A_{ij}
$$

| Symbol | Meaning |
|---|---|
| $A$ | Input matrix, $M\times N$ float32, row-major |
| $C$ | Output matrix, same shape |
| $x$ | One input element $A_{ij}$ |
| $\sqrt{2/\pi}$ | $\approx 0.7978845608$ (`kSqrt2OverPi`) |
| $0.044715$ | Cubic coefficient fitted so that the tanh form matches $\Phi$ (`kCubic`) |

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

The code evaluates the polynomial as `x + kCubic * x * x * x` and calls `tanhf` (full-precision; `__tanhf` or `tanh.approx` would be faster but less accurate than the tolerance allows near 0).

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

- **Which GELU**: the erf form and the tanh form differ by up to about
  $10^{-3}$, larger than the tolerance. Use the tanh form here (and the erf
  form when a problem asks for exact GELU).
- **Fast-math `tanh`** (`-use_fast_math`) loses relative accuracy for small
  arguments.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [ELU](../elu/), [Swish](../swish/), [Sigmoid](../sigmoid/), LeetGPU [GEGLU](../../leetgpu/065-geglu/).

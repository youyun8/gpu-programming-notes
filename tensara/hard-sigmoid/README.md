---
title: Hard Sigmoid
platform: Tensara
upstream: hard-sigmoid
url: https://tensara.org/problems/hard-sigmoid
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# Hard Sigmoid

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/hard-sigmoid)

## Problem

Apply the piecewise-linear "hard" sigmoid elementwise to an $M\times N$
float32 matrix, matching `F.hardsigmoid`. Test matrices go from $4096\times4096$ to $8192\times8192$ (up to 67 M elements). The check is
`rtol = 1e-4`, `atol = 6e-5`.

## Formulation

$$
C_{ij} = \operatorname{hsig}(A_{ij}), \qquad
\operatorname{hsig}(x) = \begin{cases} 0, & x \le -3 \\ \dfrac{x}{6} + \dfrac{1}{2}, & -3 < x < 3 \\ 1, & x \ge 3 \end{cases}
= \min\Bigl(1,\ \max\bigl(0,\ \tfrac{x}{6} + \tfrac{1}{2}\bigr)\Bigr)
$$

| Symbol | Meaning |
|---|---|
| $A$ | input matrix, $M\times N$ float32, row-major |
| $C$ | output matrix, same shape |
| $x$ | one input element $A_{ij}$ |
| $\operatorname{hsig}$ | hard sigmoid: a linear ramp from $(-3, 0)$ to $(3, 1)$ |

It is the linear interpolation of the logistic sigmoid's saturation points,
cheap on hardware without a fast exponential (mobile NPUs).

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

The code computes `fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f)`: one division (or a multiply by the compiler's reciprocal when allowed), one add and two min/max instructions.

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

- **Slope 1/6 and offset 1/2**: some frameworks (Keras) use $0.2x + 0.5$
  with breakpoints $\pm2.5$; PyTorch uses $x/6 + 1/2$.
- `x / 6.0f` and `x * (1.0f / 6.0f)` can differ by 1 ulp; both are within
  tolerance.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Sigmoid](../sigmoid/), [Swish](../swish/), [ReLU](../relu/).

---
title: SELU
platform: Tensara
upstream: selu
url: https://tensara.org/problems/selu
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# SELU

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/selu)

## Problem

Apply the Scaled Exponential Linear Unit elementwise to an $M\times N$
float32 matrix, matching `torch.selu`. Test matrices go from $4096\times4096$ to $8192\times8192$ (up to 67 M elements). The check is `rtol = 1e-4`,
`atol = 8e-5`.

## Formulation

$$
C_{ij} = \lambda \begin{cases} x, & x > 0 \\ \alpha\,(e^{x} - 1), & x \le 0 \end{cases}, \qquad x = A_{ij}
$$

| Symbol | Meaning |
|---|---|
| $A$ | input matrix, $M\times N$ float32, row-major |
| $C$ | output matrix, same shape |
| $x$ | one input element $A_{ij}$ |
| $\alpha$ | $1.6732632423543772$ (`kAlpha`) |
| $\lambda$ | $1.0507009873554805$ (`kScale`) |

The constants are the fixed point derived by Klambauer et al.
(*Self-Normalizing Neural Networks*, 2017): if inputs have mean 0 and
variance 1, so do the outputs,

$$
\mathbb{E}[\operatorname{SELU}(z)] = 0, \qquad \operatorname{Var}[\operatorname{SELU}(z)] = 1, \qquad z \sim \mathcal{N}(0, 1)
$$

| Symbol | Meaning |
|---|---|
| $z$ | a standard normal random variable |
| $\mathbb{E}, \operatorname{Var}$ | expectation and variance |

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

The code is the ELU kernel with the scale applied outside the select: `kScale * (x > 0 ? x : kAlpha * expm1f(x))`.

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

- **Constants to full float precision**: truncated values such as 1.67
  and 1.05 miss the tolerance.
- **`expm1f`** instead of `expf(x) - 1` near zero, as in [ELU](../elu/).

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [ELU](../elu/), [Leaky ReLU](../leaky-relu/), [GELU](../gelu/).

---
title: ReLU
platform: LeetGPU
upstream: easy/21_relu
url: https://leetgpu.com/challenges/relu
difficulty: easy
tags: [elementwise, activation, vectorized, memory-bound]
status: solved
---

# ReLU

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/relu)

## Problem

Apply the rectified linear unit elementwise to $N$ float32 values
($1 \le N \le 10^8$; benchmark $N = 2.5\times10^7$). ReLU is the default
nonlinearity in CNNs and MLPs. As a kernel it is the same pattern as
[Vector Addition](../001-vector-add/), with one input stream instead of two.

## Formulation

$$
y_i = \operatorname{ReLU}(x_i) = \max(0, x_i) =
\begin{cases} x_i, & x_i > 0 \\ 0, & x_i \le 0 \end{cases}
$$

| Symbol | Meaning |
|---|---|
| $N$ | number of elements |
| $x_i$ | input value (float32) |
| $y_i$ | output value (float32) |

Its derivative (needed for backprop, not here) is the step function
$\mathbb 1[x > 0]$.

## Approach

- **Vectorised body.** Thread $t < \lfloor N/4\rfloor$ loads one `float4`,
  applies `fmaxf(v, 0.0f)` to each lane, and stores one `float4`. `fmaxf`
  is a single `FMNMX` instruction, so there is no branch and no divergence.
- **Scalar tail.** Threads $t < N \bmod 4$ also process element
  $4\lfloor N/4\rfloor + t$.
- **Grid.** $\lfloor (\lfloor N/4\rfloor + 256)/256 \rfloor$ blocks, enough
  to cover the vector part and give the tail threads a home even when
  $N < 4$.

## Cost Analysis

$$
Q = 8N \ \text{bytes}, \qquad W = N, \qquad I = \frac18\ \text{FLOP/byte}, \qquad T_{\min} = \frac{8N}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM traffic: read $x$, write $y$ |
| $W$ | one max per element |
| $I$ | arithmetic intensity |
| $\beta$ | DRAM bandwidth |

Benchmark: $Q = 200$ MB, so $T_{\min} \approx 100\ \mu s$ at 2 TB/s. In a
real network, ReLU is almost always **fused** into the preceding GEMM or
convolution epilogue (see Tensara [GEMM + ReLU](../../tensara/gemm-relu/)).
That saves this entire round trip through DRAM.

## Pitfalls

- **NaN handling.** `fmaxf(NaN, 0) = 0`, but `torch.relu(NaN) = NaN`. The
  test inputs contain no NaNs. A NaN-propagating variant would be
  `x > 0 ? x : (x != x ? x : 0)`.
- **$-0.0$.** `fmaxf(-0.0f, 0.0f)` may return either zero. Both compare
  equal, so the check passes.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md), including
$N = 1, 2, 3$ (tail-only).

## Related

- [Leaky ReLU](../023-leaky-relu/), [Sigmoid](../068-sigmoid/), [SiLU](../052-silu/).
- Tensara [ReLU](../../tensara/relu/).

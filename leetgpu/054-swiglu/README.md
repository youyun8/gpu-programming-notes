---
title: Swish-Gated Linear Unit
platform: LeetGPU
upstream: easy/54_swiglu
url: https://leetgpu.com/challenges/swish-gated-linear-unit
difficulty: easy
tags: [elementwise, activation, gated]
status: solved
---

# Swish-Gated Linear Unit

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/swish-gated-linear-unit)

## Problem

The SwiGLU gate on a 1-D vector: split the length-$N$ float32 input into
halves $\mathbf x_1$ (first $N/2$) and $\mathbf x_2$ (second $N/2$), and
output $\operatorname{SiLU}(\mathbf x_1)\odot\mathbf x_2$ of length $N/2$
($N \le 10^5$, even; values in $[-100, 100]$; `atol = 1e-4`, `rtol = 1e-5`).
In an LLM's MLP, $\mathbf x_1$ and $\mathbf x_2$ are the "gate" and "up"
projections.

## Formulation

$$
y_i = \operatorname{SiLU}(x_i)\cdot x_{i + N/2} = \frac{x_i}{1 + e^{-x_i}}\cdot x_{i + N/2}, \qquad 0 \le i < N/2
$$

| Symbol | Meaning |
|---|---|
| $N$ | Input length (even) |
| $x_i$ | First half: the gate input, $0 \le i < N/2$ |
| $x_{i+N/2}$ | Second half: the value being gated |
| SiLU | $x\,\sigma(x)$; see [SiLU](../052-silu/) |
| $y_i$ | Output, length $N/2$ |

Because SiLU is smooth and non-monotonic, this "Swish gate" trains better
than a ReLU gate. It is the activation of LLaMA, Mistral and PaLM.

## Approach

One thread per output $i$:
- loads $x_i$ and $x_{i+N/2}$. For a warp, both are contiguous 128-byte
  segments, i.e. two coalesced streams;
- computes `x1 / (1 + expf(-x1)) * x2`.

If $N = 0$ the launch is skipped (0 blocks is an invalid configuration).

## Cost Analysis

$$
Q = 4N + 2N = 6N \ \text{bytes}, \qquad W \approx \tfrac{N}{2}\,(c_{\exp} + c_{\div} + 2)
$$

| Symbol | Meaning |
|---|---|
| $Q$ | Bytes: read $N$ floats, write $N/2$ floats |
| $W$ | Per output: one `expf`, one division, one add and one multiply |
| $c_{\exp},\ c_{\div}$ | Instruction costs of accurate exp and division (≈ 10–20 each) |

At $N = 10^5$ the kernel is launch-bound. In a real MLP this gate is fused
into the epilogue of the gate/up GEMM (see [SwiGLU MLP Block](../084-swiglu-mlp-block/)).

## Pitfalls

- **Halves vs. interleaving.** `chunk(2)` splits into contiguous halves. It
  does *not* pair even and odd elements.
- **Output length** is $N/2$.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md), including
$N = 2$.

## Related

- [SiLU](../052-silu/), [GEGLU](../065-geglu/), [SwiGLU MLP Block](../084-swiglu-mlp-block/).

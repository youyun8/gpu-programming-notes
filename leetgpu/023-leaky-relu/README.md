---
title: Leaky ReLU
platform: LeetGPU
upstream: easy/23_leaky_relu
url: https://leetgpu.com/challenges/leaky-relu
difficulty: easy
tags: [elementwise, activation, vectorized, memory-bound]
status: solved
---

# Leaky ReLU

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/leaky-relu)

## Problem

Leaky ReLU with slope $\alpha = 0.01$ on $N$ float32 values
($1 \le N \le 10^8$, $\lvert x_i\rvert \le 1000$; benchmark $N = 5\times10^7$;
tolerance `1e-6`). Unlike ReLU it keeps a small gradient for negative
inputs, which avoids "dead" units during training.

## Formulation

$$
y_i = \operatorname{LeakyReLU}(x_i) =
\begin{cases} x_i, & x_i > 0 \\ \alpha\, x_i, & x_i \le 0 \end{cases}
\qquad \alpha = 0.01
$$

| Symbol | Meaning |
|---|---|
| $N$ | Number of elements |
| $x_i,\ y_i$ | Input and output values (float32) |
| $\alpha$ | negative-side slope |

Equivalently $y = \max(x, \alpha x)$ for $0 < \alpha < 1$.

## Approach

The same vectorised template as [ReLU](../021-relu/): one `float4` per
thread plus a scalar tail. The ternary `x > 0 ? x : 0.01f * x` compiles to a
multiply and a select (`FSEL`), so there is no divergent branch.

The product `0.01f * x` is a single float32 multiply. It is rounded exactly
as PyTorch's `alpha * input` in float32, which matters for the tight `1e-6`
tolerance.

## Cost Analysis

$$
Q = 8N \ \text{bytes}, \qquad T_{\min} = \frac{8N}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM traffic (read input, write output) |
| $\beta$ | DRAM bandwidth |

Benchmark: $Q = 400$ MB, so $T_{\min} \approx 200\ \mu s$ at 2 TB/s.

## Pitfalls

- **Double-precision constant.** Writing `0.01 * x` (a double literal)
  promotes to float64, which is slow on consumer GPUs and can round
  differently. Use `0.01f`.
- **Boundary.** $x = 0$ belongs to the $\alpha x$ branch, which gives 0 either way.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-6`.

## Related

- [ReLU](../021-relu/), Tensara [Leaky ReLU](../../tensara/leaky-relu/), [ELU](../../tensara/elu/).

---
title: Mean Squared Error
platform: LeetGPU
upstream: medium/27_mean_squared_error
url: https://leetgpu.com/challenges/mean-squared-error
difficulty: medium
tags: [reduction, two-pass, loss]
status: solved
---

# Mean Squared Error

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/mean-squared-error)

## Problem

Mean squared error between two float32 arrays of length $N$
($1 \le N \le 10^8$, values in $[-1000, 1000]$; benchmark $N = 5\times10^7$;
tolerance `1e-5`), written to `mse[0]`. It is a reduction with a
transform on the way in: square the difference.

## Formulation

$$
\operatorname{MSE} = \frac{1}{N}\sum_{i=0}^{N-1} \bigl(p_i - t_i\bigr)^2
$$

| Symbol | Meaning |
|---|---|
| $N$ | number of elements |
| $p_i$ | prediction (float32) |
| $t_i$ | target (float32) |
| $\operatorname{MSE}$ | result, stored as float32 |

Each squared term can reach $(2000)^2 = 4\times10^6$. The sum of $10^8$ such
terms can reach $4\times10^{14}$, far beyond float32's 24-bit mantissa for
exact accumulation. Only per-thread partials are accumulated in float32
(each thread sums a few hundred terms); the cross-thread levels use
float64.

## Approach

The same two-pass structure as [Reduction](../004-reduction/):

1. **`partialSquares`**: grid-stride over `float4` pairs, accumulating
   $(a-b)^2$ for 4 lanes per iteration (pairwise-added to shorten the
   dependency chain), then a scalar tail. Block reduction in float64 gives
   one partial per block.
2. **`finalMean`**: one block sums the partials in float64, divides by $N$,
   and rounds to float32.

## Cost analysis

$$
Q = 8N\ \text{bytes}, \qquad W = 3N, \qquad I = \frac{3}{8}\ \text{FLOP/byte}, \qquad T_{\min} = \frac{8N}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes (read both arrays once) |
| $W$ | FLOPs: subtract, multiply, add per element |
| $I$ | arithmetic intensity |
| $\beta$ | DRAM bandwidth |

Benchmark: 400 MB, so ≈ 200 µs at 2 TB/s. The kernel is memory-bound.

## Pitfalls

- **Dividing in float32 at the end.** Fine. Summing in float32 at the upper
  levels is not (see above).
- **Grid of 0 blocks** for $N < 4$: clamped to 1.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-5`.

## Related

- [Reduction](../004-reduction/), [Dot Product](../017-dot-product/).
- Tensara [MSE Loss](../../tensara/mse-loss/), [Huber Loss](../../tensara/huber-loss/).

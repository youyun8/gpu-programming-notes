---
title: Frobenius Normalization
platform: Tensara
upstream: frobenius-norm
url: https://tensara.org/problems/frobenius-norm
difficulty: easy
tags: [normalization, reduction, fp64-accumulation, grid-reduction]
status: solved
---

# Frobenius Normalization

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/frobenius-norm)

## Problem

Divide every element of a float32 tensor of arbitrary shape by the
tensor's Frobenius norm. Only the total element count $n$ is passed; the
tests have 4 M to 33 M elements (for example $(4, 16, 32, 128, 128)$). The
check is `rtol = 3e-3`, `atol = 1e-6`.

## Formulation

$$
\lVert X \rVert_F = \sqrt{\sum_{k=0}^{n-1} x_k^2}, \qquad y_k = \frac{x_k}{\lVert X \rVert_F}
$$

| Symbol | Meaning |
|---|---|
| $X$ | The input tensor, viewed as a flat vector of $n$ elements |
| $x_k, y_k$ | Input and output element $k$ |
| $\lVert X\rVert_F$ | Frobenius norm: the L2 norm of the flattened tensor |

The global sum is split into per-block partial sums:

$$
S = \sum_{b=0}^{G-1} S_b, \qquad S_b = \sum_{k \in \mathcal{K}_b} x_k^2, \qquad r = \frac{1}{\sqrt{S}}
$$

| Symbol | Meaning |
|---|---|
| $G$ | Number of blocks in the first kernel ($\le 1024$) |
| $\mathcal{K}_b$ | The indices visited by block $b$'s grid-stride loop |
| $S_b$ | Block partial (stored as `double`) |
| $r$ | Reciprocal norm, `g_inv_norm` |

## Approach

1. **`sumSquares`**: grid-stride loop over `float4` vectors; each thread
   keeps a float running sum of $x^2$ over at most a few hundred elements,
   then the block reduces the per-thread sums in `double` (warp shuffles
   plus shared memory) and writes $S_b$ to a device array.
2. **`finishNorm`**: one block adds the $G$ partials in `double` and stores
   $r = 1/\sqrt{S}$ as a float.
3. **`scale`**: elementwise $y_k = x_k\,r$.

Separate launches give a device-wide barrier for free; no atomics or
cooperative groups are needed, and the result is deterministic (the same
partition every run).

## Cost Analysis

$$
Q = 4n\ (\text{pass 1}) + 8n\ (\text{pass 3}) = 12n\ \text{bytes}, \qquad T_{\min} = \frac{12n}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes |
| $\beta$ | DRAM bandwidth |

At $n = 33.5$ M: 403 MB, about 0.2 ms at 2 TB/s. Both passes are
unavoidable: no output can be written before the whole sum is known
(unless the tensor fits in on-chip memory, which it does not).

## Pitfalls

- **Precision**: a single float accumulator over $3\times10^7$ squares loses
  several digits; fp64 partials keep the norm exact to fp32 precision.
- **Multiply by $r$** instead of dividing by the norm: at most 1 ulp
  difference, well inside `rtol = 3e-3`.
- **Arbitrary shape**: only $n$ matters, so the kernel is shape-agnostic.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [L2 Norm](../l2-norm/) (per row), [MSE Loss](../mse-loss/) (same
  two-level reduction), LeetGPU [Reduction](../../leetgpu/004-reduction/).

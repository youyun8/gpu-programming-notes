---
title: Mean Squared Error Loss
platform: Tensara
upstream: mse-loss
url: https://tensara.org/problems/mse-loss
difficulty: easy
tags: [loss, reduction, fp64-accumulation, grid-reduction]
status: solved
---

# Mean Squared Error Loss

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/mse-loss)

## Problem

Mean squared error between two float32 tensors of arbitrary shape, from
$4096^2$ up to $8192^2$ and $512^3$ (134 M elements), returning one
scalar. The shape arrives as an array `shape` of `ndim` 64-bit sizes. The
check is `rtol = 5e-5`, `atol = 1e-4`.

## Formulation

$$
n = \prod_{k=0}^{\text{ndim}-1} S_k, \qquad
\text{MSE} = \frac{1}{n}\sum_{i=0}^{n-1} \bigl(x_i - y_i\bigr)^2
$$

| Symbol | Meaning |
|---|---|
| $S_k$ | size of dimension $k$ |
| $n$ | total number of elements |
| $x_i, y_i$ | prediction and target, flattened |
| MSE | the scalar output |

The sum uses a two-level partition, as in [Frobenius Norm](../frobenius-norm/):

$$
\text{MSE} = \frac{1}{n}\sum_{b=0}^{G-1} P_b, \qquad P_b = \sum_{i \in \mathcal{K}_b} (x_i - y_i)^2
$$

| Symbol | Meaning |
|---|---|
| $G$ | number of blocks of the first kernel ($\le 1024$) |
| $\mathcal{K}_b$ | the indices block $b$ visits |
| $P_b$ | block partial sum (fp64) |

## Approach

1. **Element count on the host**: `shape` may be a host or a device pointer,
   so it is copied with `cudaMemcpyDefault` (unified virtual addressing
   picks the direction) and multiplied out.
2. **`squaredDiffs`**: grid-stride loop, per-thread float accumulator,
   block reduction in `double`, one partial per block.
3. **`finalize`**: one block sums the partials in `double` and writes
   $\sum/n$ as a float.

The result is deterministic, unlike an `atomicAdd` into one float.

## Cost Analysis

$$
Q = 8n\ \text{bytes}, \qquad T_{\min} = \frac{8n}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes: both tensors read once, output negligible |
| $\beta$ | DRAM bandwidth |

At $n = 2^{27}$ ($512^3$): 1.07 GB, about 0.54 ms at 2 TB/s.

## Pitfalls

- **Precision**: $1.3\times10^8$ squared residuals summed in fp32 would lose
  the `rtol = 5e-5` budget; fp64 partials keep it.
- **`shape` pointer kind**: dereferencing a device pointer on the host
  crashes; `cudaMemcpyDefault` handles both.
- **Scalar output**: write exactly one float.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Huber Loss](../huber-loss/), [Frobenius Norm](../frobenius-norm/),
  LeetGPU [Mean Squared Error](../../leetgpu/027-mean-squared-error/).

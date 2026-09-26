---
title: L2 Normalization
platform: Tensara
upstream: l2-norm
url: https://tensara.org/problems/l2-norm
difficulty: easy
tags: [normalization, reduction, row-per-block]
status: solved
---

# L2 Normalization

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/l2-norm)

## Problem

Row-wise L2 normalization of a $B\times D$ float32 matrix
($B = 128, 256$; $D = 4096 \dots 16384$): divide each row by its Euclidean
norm plus $\epsilon = 10^{-10}$. The check is `rtol = 1e-4`, `atol = 1e-6`.

## Formulation

$$
n_b = \sqrt{\sum_{d=0}^{D-1} x_{bd}^2}, \qquad y_{bd} = \frac{x_{bd}}{n_b + \epsilon}
$$

| Symbol | Meaning |
|---|---|
| $B, D$ | number of rows, row length |
| $x_{bd}, y_{bd}$ | input and output element |
| $n_b$ | L2 norm of row $b$ |
| $\epsilon$ | $10^{-10}$, added after the square root |

Each output row is then a unit vector: $\sum_d y_{bd}^2 \approx 1$.

## Approach

Same structure as [L1 Norm](../l1-norm/): one 256-thread block per row;
pass 1 accumulates $x^2$ per thread and reduces across the block (warp
shuffles plus shared memory); pass 2 multiplies by
$1/(\sqrt{s} + \epsilon)$, re-reading the row from L2.

## Cost Analysis

$$
Q_{\text{DRAM}} \approx 8BD\ \text{bytes}, \qquad W = 3BD\ \text{flops}
$$

| Symbol | Meaning |
|---|---|
| $Q_{\text{DRAM}}$ | one read (the second read hits L2) and one write |
| $W$ | one FMA for $x^2$ and one multiply for the scale |

## Pitfalls

- **Tight `atol = 1e-6`**: outputs are $O(1/\sqrt{D}) \approx 0.01$, so a
  relative error of $10^{-4}$ matters; the float running sum over 16 K
  elements (64 per thread before the tree) is accurate enough.
- **$\epsilon$ after the square root**, not inside it (which would be RMS
  Norm's convention).

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [L1 Norm](../l1-norm/), [Frobenius Norm](../frobenius-norm/),
  [Cosine Similarity](../cosine-similarity/), [RMS Norm](../rms-norm/).

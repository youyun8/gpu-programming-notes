---
title: Triplet Margin Loss
platform: Tensara
upstream: triplet-margin
url: https://tensara.org/problems/triplet-margin
difficulty: medium
tags: [loss, reduction, row-per-block, fp64-accumulation]
status: solved
---

# Triplet Margin Loss

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/triplet-margin)

## Problem

Triplet margin loss for a batch of $B$ triplets (anchor, positive,
negative) of embedding size $E$ (up to $B = 1024$, $E = 16384$), with a
runtime margin, matching `nn.TripletMarginLoss(margin)` (L2 distance,
mean reduction). The output is one scalar. The check is
`rtol = atol = 6e-4`.

## Formulation

$$
d(\mathbf{x}, \mathbf{y}) = \Bigl\lVert \mathbf{x} - \mathbf{y} + \epsilon\mathbf{1} \Bigr\rVert_2 = \sqrt{\sum_{e=0}^{E-1} (x_e - y_e + \epsilon)^2}
$$

$$
\ell_i = \max\bigl(0,\ d(\mathbf{a}_i, \mathbf{p}_i) - d(\mathbf{a}_i, \mathbf{n}_i) + m\bigr), \qquad
\mathcal{L} = \frac{1}{B}\sum_{i=0}^{B-1} \ell_i
$$

| Symbol | Meaning |
|---|---|
| $\mathbf{a}_i, \mathbf{p}_i, \mathbf{n}_i$ | Anchor, positive and negative embeddings of triplet $i$ (rows of $B\times E$ matrices) |
| $d$ | `torch.pairwise_distance`: L2 norm of the difference plus $\epsilon$ in every component |
| $\epsilon$ | $10^{-6}$ (PyTorch's default) |
| $m$ | Margin |
| $\ell_i$ | per-triplet hinge |
| $\mathcal{L}$ | Scalar output |

The loss is zero once the negative is farther from the anchor than the
positive by at least $m$.

## Approach

1. **One block per triplet** (256 threads). Each thread strides the $E$
   columns and accumulates both squared distances,
   $(a - p + \epsilon)^2$ and $(a - n + \epsilon)^2$, in one pass: the
   anchor row is read once, all three rows coalesced.
2. Two block reductions give $d_{ap}$ and $d_{an}$; thread 0 writes
   $\ell_i$ to a small temporary buffer.
3. **A final single block** sums the $B$ hinge values in `double` and
   writes $\mathcal{L}$.

## Cost Analysis

$$
Q = 12BE\ \text{bytes}, \qquad W = 6BE\ \text{flops}, \qquad T_{\min} = \frac{12BE}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes: three $B\times E$ matrices read once |
| $W$ | Two subtract-add-FMA chains per column |
| $\beta$ | DRAM bandwidth |

At $B = 256$, $E = 16384$: 50 MB, ~25 µs at 2 TB/s.

## Pitfalls

- **The $\epsilon$ is inside the difference**, not added to the norm;
  omitting it changes results by about $\epsilon\sqrt{E}$ relative terms,
  which matters for close pairs.
- **Mean over the batch**, not the sum.
- **Hinge per triplet before averaging**.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Hinge Loss](../hinge-loss/), [Cosine Similarity](../cosine-similarity/), [L2 Norm](../l2-norm/).

---
title: Huber Loss
platform: Tensara
upstream: huber-loss
url: https://tensara.org/problems/huber-loss
difficulty: easy
tags: [loss, elementwise]
status: solved
---

# Huber Loss

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/huber-loss)

## Problem

Element-wise Smooth L1 loss (Huber loss with $\delta = 1$) between
predictions and targets of length $N$ (1 M … 67 M), matching
`F.smooth_l1_loss(p, t, reduction='none', beta=1.0)`. The check is
`rtol = 2e-4`, `atol = 1e-4`.

## Formulation

$$
d_i = x_i - y_i, \qquad
z_i = \begin{cases} \dfrac{d_i^2}{2\beta}, & \lvert d_i \rvert < \beta \\[4pt] \lvert d_i \rvert - \dfrac{\beta}{2}, & \text{otherwise} \end{cases}, \qquad \beta = 1
$$

| Symbol | Meaning |
|---|---|
| $x_i, y_i$ | prediction and target |
| $d_i$ | residual |
| $\beta$ | transition point between the quadratic and linear regimes |
| $z_i$ | per-element loss, the output |

Both pieces meet with equal value and slope at $\lvert d\rvert = \beta$:

$$
\frac{\beta^2}{2\beta} = \beta - \frac{\beta}{2} = \frac{\beta}{2}, \qquad
\frac{d}{dd}\Bigl(\frac{d^2}{2\beta}\Bigr)\Big|_{d=\beta} = 1 = \frac{d}{dd}\bigl(d - \tfrac{\beta}{2}\bigr)
$$

| Symbol | Meaning |
|---|---|
| $\frac{d}{dd}$ | derivative with respect to the residual |

So the loss is quadratic (like MSE) for small errors and linear (like L1,
robust to outliers) for large ones.

## Approach

Grid-stride elementwise map over two inputs:
`a = fabsf(d); out = a < 1 ? 0.5f*d*d : a - 0.5f`. The branch is a
select, so there is no divergence.

## Cost Analysis

$$
Q = 12N\ \text{bytes}, \qquad T_{\min} = \frac{12N}{\beta_{\text{mem}}}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes: two inputs, one output |
| $\beta_{\text{mem}}$ | DRAM bandwidth (named to avoid clashing with the loss parameter $\beta$) |

At $N = 2^{26}$: 805 MB, about 0.4 ms at 2 TB/s.

## Pitfalls

- **Strict inequality** $\lvert d\rvert < 1$ for the quadratic branch
  (both branches agree at 1, so this only matters for exactness).
- **Element-wise output**, not the mean.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [MSE Loss](../mse-loss/), [Hinge Loss](../hinge-loss/), [KL Loss](../kl-loss/).

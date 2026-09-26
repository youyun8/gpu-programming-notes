---
title: RMS Normalization
platform: Tensara
upstream: rms-norm
url: https://tensara.org/problems/rms-norm
difficulty: easy
tags: [normalization, reduction, row-per-block]
status: solved
---

# RMS Normalization

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/rms-norm)

## Problem

RMS normalization of each row of a $B\times N$ float32 matrix (no weight),
$\epsilon = 10^{-5}$: shapes $(1024, 1024)$ to $(512, 16384)$. The check is
`rtol = 2e-4`, `atol = 1e-4`.

## Formulation

$$
\operatorname{RMS}_b = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x_{bn}^2 + \epsilon}, \qquad
y_{bn} = \frac{x_{bn}}{\operatorname{RMS}_b}
$$

| Symbol | Meaning |
|---|---|
| $B$ | number of rows (samples) |
| $N$ | features per row |
| $x_{bn}, y_{bn}$ | input and output element |
| $\operatorname{RMS}_b$ | root mean square of row $b$, with $\epsilon$ inside the square root |
| $\epsilon$ | $10^{-5}$ |

Compared with [Layer Norm](../layer-norm/), RMSNorm drops the mean
subtraction: it rescales without re-centering, which saves one reduction
and works just as well in transformers (LLaMA, T5).

## Approach

One block of 256 threads per row. Pass 1: each thread accumulates $x^2$
over its strided elements, and a block reduction (warp shuffles plus one
shared-memory hop) gives $\sum x^2$. Every thread computes
$r_b = 1/\sqrt{\sum x^2/N + \epsilon}$. Pass 2 writes $y = x\,r_b$,
re-reading the row from L1/L2.

## Cost analysis

$$
Q_{\text{DRAM}} \approx 8BN\ \text{bytes}, \qquad W = 3BN\ \text{flops}, \qquad T_{\min} = \frac{8BN}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q_{\text{DRAM}}$ | one read (the re-read hits cache) and one write |
| $W$ | an FMA for $x^2$ and a multiply for the scale |
| $\beta$ | DRAM bandwidth |

At $(2048, 8192)$: 134 MB, about 67 µs at 2 TB/s.

## Pitfalls

- **$\epsilon$ inside the root** (added to the mean square), not added to
  the RMS as in [L2 Norm](../l2-norm/).
- **Divide by $N$** (mean), not by $N - 1$.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Layer Norm](../layer-norm/), [L2 Norm](../l2-norm/),
  LeetGPU [RMS Normalization](../../leetgpu/050-rms-normalization/),
  LeetGPU [Fused Residual Add + RMSNorm](../../leetgpu/083-fused-residual-add-rms-norm/).

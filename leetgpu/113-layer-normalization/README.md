---
title: Layer Normalization
platform: LeetGPU
upstream: medium/113_layer_normalization
url: https://leetgpu.com/challenges/layer-normalization
difficulty: medium
tags: [normalization, row-reduction, warp-per-row, transformer]
status: solved
---

# Layer Normalization

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/layer-normalization)

## Problem

LayerNorm forward on an $N\times C$ float32 matrix: each row (sample or
token) is normalised over its $C$ features, then scaled and shifted by
per-feature weight and bias ($N \le 65\,536$, $C \le 4096$,
$\varepsilon = 10^{-5}$; benchmark $N = 65\,536$, $C = 512$; tolerance `1e-4`).

## Formulation

$$
\mu_i = \frac1C\sum_{j=0}^{C-1} x_{ij}, \qquad
\sigma^2_i = \frac1C\sum_{j=0}^{C-1}(x_{ij} - \mu_i)^2, \qquad
y_{ij} = w_j\,\frac{x_{ij} - \mu_i}{\sqrt{\sigma^2_i + \varepsilon}} + b_j
$$

| Symbol | Meaning |
|---|---|
| $N,\ C$ | Rows and features |
| $x_{ij}$ | Input |
| $\mu_i,\ \sigma^2_i$ | Row mean and biased variance |
| $\varepsilon$ | Stability constant |
| $w_j,\ b_j$ | per-feature scale and shift (shared by all rows) |
| $y_{ij}$ | Output |

### Why Two Passes for the Variance

The one-pass formula $\sigma^2 = E[x^2] - \mu^2$ subtracts two large, nearly
equal numbers when $\lvert\mu\rvert \gg \sigma$. The relative error is about
$u\cdot\mu^2/\sigma^2$, where $u$ is the float32 unit roundoff. With inputs
up to 100 and a small spread, that destroys the result in float32. The
centred form $\frac1C\sum(x - \mu)^2$ has no cancellation.

## Approach

**One warp per row** (8 rows per 256-thread block):

1. Pass 1: lanes stride the row (coalesced), sum, and do a butterfly
   reduction, so $\mu$ is known in every lane.
2. Pass 2: $\sum (x - \mu)^2$ is reduced the same way, then
   $\text{rstd} = \texttt{rsqrtf}(\sigma^2 + \varepsilon)$.
3. Pass 3: write $w_j\bigl((x - \mu)\,\text{rstd}\bigr) + b_j$.

The row ($\le 16$ KB) stays in L1 across the three passes, so DRAM traffic is
about one read and one write. A warp per row avoids shared memory and
`__syncthreads()` entirely, and $N = 65\,536$ rows provide ample parallelism.

## Cost Analysis

$$
Q \approx 8NC + 8C\ \text{bytes}, \qquad W \approx 8NC, \qquad T_{\min} = \frac{8NC}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes: read $x$ once (further passes hit L1), write $y$, plus the weights |
| $W$ | FLOPs |
| $\beta$ | DRAM bandwidth |

Benchmark: 268 MB, i.e. ≈ 134 µs at 2 TB/s.

## Pitfalls

- **Unbiased variance** ($C - 1$) is wrong here.
- **Very large $C$** (e.g. 16K+): a warp per row loses L1 residency and has
  too little parallelism per row. Use a block per row instead.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-4`,
including $C = 1$ (output = bias).

## Related

- [RMS Normalization](../050-rms-normalization/), [Group Normalization](../105-group-normalization/),
  [Batch Normalization](../040-batch-normalization/), [Token Embedding](../106-token-embedding-layer/).
  Tensara [Layer Norm](../../tensara/layer-norm/).

---
title: Group Normalization
platform: LeetGPU
upstream: medium/105_group_normalization
url: https://leetgpu.com/challenges/group-normalization
difficulty: medium
tags: [normalization, row-reduction, cnn]
status: solved
---

# Group Normalization

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/group-normalization)

## Problem

Group Normalization on an NCHW tensor. The $C$ channels are split into $G$
contiguous groups, and each (sample, group) is normalised over its
$(C/G)\cdot H\cdot W$ elements, then scaled and shifted per channel
($N \le 32$, $C \le 1024$; tolerance `1e-4`). GroupNorm is the
normalisation of Stable Diffusion's U-Net and many ResNets. Unlike BatchNorm,
it does not depend on the batch size. $G = 1$ gives LayerNorm, and $G = C$
gives InstanceNorm.

## Formulation

$$
\mathcal S_{n,g} = \Bigl\{(c, h, w) : g\tfrac{C}{G} \le c < (g+1)\tfrac{C}{G}\Bigr\}, \qquad
\mu_{n,g} = \frac{1}{\lvert\mathcal S\rvert}\sum_{\mathcal S_{n,g}} x_{n,c,h,w}, \qquad
\sigma^2_{n,g} = \frac{1}{\lvert\mathcal S\rvert}\sum_{\mathcal S_{n,g}}\bigl(x_{n,c,h,w} - \mu_{n,g}\bigr)^2
$$

$$
y_{n,c,h,w} = \gamma_c\,\frac{x_{n,c,h,w} - \mu_{n,g(c)}}{\sqrt{\sigma^2_{n,g(c)} + \varepsilon}} + \beta_c, \qquad g(c) = \Bigl\lfloor\frac{cG}{C}\Bigr\rfloor
$$

| Symbol | Meaning |
|---|---|
| $N,\ C,\ H,\ W$ | Batch, channels, height, width |
| $G$ | Number of groups ($C$ divisible by $G$) |
| $\mathcal S_{n,g}$ | Elements of group $g$ in sample $n$; $\lvert\mathcal S\rvert = (C/G)HW$ |
| $\mu_{n,g},\ \sigma^2_{n,g}$ | Group mean and biased variance |
| $g(c)$ | Group of channel $c$ |
| $\gamma_c,\ \beta_c$ | per-channel affine parameters |
| $\varepsilon$ | Stability constant |

**Contiguity.** In NCHW layout, the channels $g\frac CG \dots (g+1)\frac CG - 1$
of sample $n$ are one contiguous block of $\lvert\mathcal S\rvert$ floats,
starting at $\bigl(nC + g\tfrac CG\bigr)HW$. Every group is a flat
contiguous row.

## Approach

**One block of 256 threads per (n, g)** (grid $N\cdot G$):

1. Grid-stride over the group's contiguous elements, accumulating $\sum x$
   and $\sum x^2$ in **float64**. A combined block reduction does both sums
   in one pass: warp shuffles, then a shared-memory hop.
2. $\mu = \sum x/\lvert\mathcal S\rvert$ and
   $\sigma^2 = \sum x^2/\lvert\mathcal S\rvert - \mu^2$, which is safe in float64
   and clamped at 0. Then $\text{rstd} = 1/\sqrt{\sigma^2 + \varepsilon}$.
3. Second pass: $y = (x - \mu)\cdot\text{rstd}\cdot\gamma_c + \beta_c$, with
   $c = g\frac CG + \lfloor i/(HW)\rfloor$.

## Cost Analysis

$$
Q = 12\,NCHW\ \text{bytes} \quad(\text{read twice, write once}), \qquad \text{parallelism} = N\cdot G\ \text{blocks}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes (the second read usually hits L2 for moderate group sizes) |
| Parallelism | Number of independent blocks |

With few groups and large $HW$ (e.g. $N = 1$, $G = 32$, $64\times64$ feature
maps), only 32 blocks run. Splitting each group over several blocks with a
two-level reduction (as in [Batch Norm](../040-batch-normalization/)) would
use more SMs.

## Pitfalls

- **Float32 $E[x^2] - \mu^2$** loses precision for large means. The float64
  accumulation avoids it.
- **Channel of an element** inside the group: $\lfloor i/HW\rfloor$, not
  $i \bmod (C/G)$.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-4`,
including $G = 1$ and $G = C$.

## Related

- [Batch Norm](../040-batch-normalization/), [Layer Norm](../113-layer-normalization/),
  [RMS Norm](../050-rms-normalization/), [DiT Block](../116-dit-block/).

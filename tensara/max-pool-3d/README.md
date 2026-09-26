---
title: 3D Max Pooling
platform: Tensara
upstream: max-pool-3d
url: https://tensara.org/problems/max-pool-3d
difficulty: hard
tags: [pooling, stencil, dilation, grid-stride]
status: solved
---

# 3D Max Pooling

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/max-pool-3d)

## Problem

3-D max pooling of a float32 tensor with a window of side $k$, stride
$S$, padding $P$ and **dilation** $\delta$, matching
`F.max_pool3d(x, k, S, P, dilation=δ)`. Padded positions act as
$-\infty$ (they never win). The input is an $H\times W\times D$ volume ($D$ contiguous). The check is `rtol = atol = 9e-5`.

## Formulation

The dilated window spans $\delta(k-1) + 1$ input positions per axis, so

$$
X_{\text{out}} = \left\lfloor \frac{X + 2P - \delta(k - 1) - 1}{S} \right\rfloor + 1
$$

| Symbol | Meaning |
|---|---|
| $X$ | input extent along one axis ($H$, $W$ or $D$) |
| $X_{\text{out}}$ | output extent along that axis |
| $k$ | window side (`kernel_size`) |
| $S$ | stride |
| $P$ | padding on each side |
| $\delta$ | dilation: distance between neighbouring window taps |

$$
\text{out}[a, b, c] = \max_{\substack{0 \le m, n, o < k \\ \text{in bounds}}} x\bigl[aS - P + m\delta,\ bS - P + n\delta,\ cS - P + o\delta\bigr]
$$

| Symbol | Meaning |
|---|---|
| $x$ | input volume, $H\times W\times D$, row-major |
| $a, b, c$ | output indices along $H, W, D$ |
| $m, n, o$ | tap indices along the three axes |

The flat output index $t$ decodes as $c = t \bmod D_{\text{out}}$,
$b = \lfloor t/D_{\text{out}}\rfloor \bmod W_{\text{out}}$,
$a = \lfloor t/(D_{\text{out}}W_{\text{out}})\rfloor$.

| Symbol | Meaning |
|---|---|
| $t$ | flat output index of a thread |

## Approach

1. **One thread per output** (grid-stride loop over the flat output
   index); consecutive threads produce consecutive outputs along the
   innermost axis, so their windows overlap and the loads are served from
   L1/L2.
2. **Padding is a bounds test**: taps outside the input are skipped, which
   is the same as treating them as $-\infty$.
3. **Accumulator** starts at $-\text{FLT\_MAX}$ and folds with `fmaxf`.
   `max` is exact, so the result is bit-identical to PyTorch.

## Cost Analysis

$$
W_{\text{ops}} = k^{3}\,H_{\text{out}}W_{\text{out}}D_{\text{out}}, \qquad Q \approx 4\,(\text{input size} + H_{\text{out}}W_{\text{out}}D_{\text{out}})\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $W_{\text{ops}}$ | comparisons |
| $Q$ | compulsory DRAM bytes, assuming overlapping windows hit in cache |
| $\beta$ | DRAM bandwidth |

With $S < k$ windows overlap and the kernel is bandwidth-bound; with
large strides each input is read once anyway.

## Pitfalls

- **Dilation in the output size**: the effective window is
  $\delta(k-1)+1$, not $k$.
- **Padding is $-\infty$, not 0**: a window that sees only negatives
  plus padding must return the largest negative, not 0.
- PyTorch requires $P \le k/2$, so every window contains at least one
  real element and the result is never $-\text{FLT\_MAX}$.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Max Pool 2D](../max-pool-2d/), [Avg Pool 3D](../avg-pool-3d/), [Conv Square 3D](../conv-square-3d/).

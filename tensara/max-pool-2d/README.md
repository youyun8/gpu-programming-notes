---
title: 2D Max Pooling
platform: Tensara
upstream: max-pool-2d
url: https://tensara.org/problems/max-pool-2d
difficulty: medium
tags: [pooling, stencil, dilation, grid-stride]
status: solved
---

# 2D Max Pooling

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/max-pool-2d)

## Problem

2-D max pooling of a float32 tensor with a window of side $k$, stride
$S$, padding $P$ and **dilation** $\delta$, matching
`F.max_pool2d(x, k, S, P, dilation=δ)`. Padded positions act as
$-\infty$ (they never win). The input is an $H\times W$ matrix. The check is `rtol = 1e-4`, `atol = 8e-5`.

## Formulation

The dilated window spans $\delta(k-1) + 1$ input positions per axis, so

$$
X_{\text{out}} = \left\lfloor \frac{X + 2P - \delta(k - 1) - 1}{S} \right\rfloor + 1
$$

| Symbol | Meaning |
|---|---|
| $X$ | input extent along one axis ($H$ or $W$) |
| $X_{\text{out}}$ | output extent along that axis |
| $k$ | window side (`kernel_size`) |
| $S$ | stride |
| $P$ | padding on each side |
| $\delta$ | dilation: distance between neighbouring window taps |

$$
\text{out}[a, b] = \max_{\substack{0 \le m, n < k \\ (r, c)\ \text{in bounds}}} x[r, c], \qquad r = aS - P + m\delta, \quad c = bS - P + n\delta
$$

| Symbol | Meaning |
|---|---|
| $x$ | input, $H\times W$, row-major |
| $a, b$ | output row and column |
| $m, n$ | tap indices along rows and columns |
| $r, c$ | input coordinates of a tap |

The flat output index $t$ decodes as $b = t \bmod W_{\text{out}}$,
$a = \lfloor t / W_{\text{out}} \rfloor$.

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
W_{\text{ops}} = k^{2}\,H_{\text{out}}W_{\text{out}}, \qquad Q \approx 4\,(\text{input size} + H_{\text{out}}W_{\text{out}})\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
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

- [Max Pool 1D](../max-pool-1d/), [Max Pool 3D](../max-pool-3d/), [Avg Pool 2D](../avg-pool-2d/), LeetGPU [2D Max Pooling](../../leetgpu/042-2d-max-pooling/).

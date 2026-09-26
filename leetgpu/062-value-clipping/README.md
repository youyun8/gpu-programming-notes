---
title: Value Clipping
platform: LeetGPU
upstream: easy/62_value_clipping
url: https://leetgpu.com/challenges/value-clipping
difficulty: easy
tags: [elementwise, clamp]
status: solved
---

# Value Clipping

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/value-clipping)

## Problem

Clip each of $N$ float32 values into $[\ell, h]$ ($N \le 10^5$,
$\ell \le h$; tolerance `1e-5`). Clipping (clamping) appears in
gradient/activation stabilisation, PPO's ratio clipping and before
quantisation.

## Formulation

$$
y_i = \operatorname{clamp}(x_i, \ell, h) = \min\bigl(\max(x_i, \ell),\ h\bigr) =
\begin{cases} \ell, & x_i < \ell \\ x_i, & \ell \le x_i \le h \\ h, & x_i > h \end{cases}
$$

| Symbol | Meaning |
|---|---|
| $N$ | Number of elements |
| $x_i,\ y_i$ | Input and output values |
| $\ell,\ h$ | Lower and upper bounds (`lo`, `hi`) |

## Approach

One thread per element: `fminf(fmaxf(x, lo), hi)`. These are two
`FMNMX` instructions, with no branches and no divergence.

## Cost Analysis

$$
Q = 8N\ \text{bytes}, \qquad T_{\min} = \frac{8N}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes (read and write each element) |
| $\beta$ | DRAM bandwidth |

At $N = 10^5$ (800 KB) the kernel is launch-bound (a few µs).

## Pitfalls

- **NaN inputs.** `fmaxf(NaN, lo) = lo`, while `torch.clamp(NaN)` returns
  NaN. The tests contain no NaNs.
- **Order of min/max** matters only if $\ell > h$, which the constraints exclude.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md), including
$\ell = h$.

## Related

- [ReLU](../021-relu/) (clamp at 0 from below), [PPO Clipped Loss](../107-ppo-clipped-surrogate-loss/).
- Tensara [Hard Sigmoid](../../tensara/hard-sigmoid/), [Threshold](../../tensara/threshold/).

---
title: Argmin Over Dimension
platform: Tensara
upstream: argmin
url: https://tensara.org/problems/argmin
difficulty: easy
tags: [reduction, argmin, strided-reduction]
status: solved
---

# Argmin Over Dimension

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/argmin)

## Problem

Index of the minimum along dimension `dim` of an $n$-D float32 tensor
(first occurrence on ties), with the reduced dimension removed. It is the
mirror image of [Argmax](../argmax/), with the same test shapes.

## Formulation

With the $(O, R, I)$ view of [Argmax](../argmax/):

$$
\text{out}[oI + i] = \min\Bigl\{\ j^\star\ :\ x[o, j^\star, i] = \min_{0\le j<R} x[o, j, i]\ \Bigr\}, \qquad
(v_1, j_1)\oplus(v_2, j_2) = \begin{cases}(v_2, j_2), & v_2 < v_1 \lor (v_2 = v_1 \land j_2 < j_1)\\(v_1, j_1), & \text{otherwise}\end{cases}
$$

| Symbol | Meaning |
|---|---|
| $O,\ R,\ I$ | outer size, reduced length, inner size (stride of the reduced axis) |
| $x[o, j, i]$ | element $\text{input}[(oR + j)I + i]$ |
| $\oplus$ | arg-min with first-index tie-break |
| out | int32 indices, $O\cdot I$ of them |

## Approach

Identical to [Argmax](../argmax/) with the comparison flipped and the
identity value set to $+\text{FLT\_MAX}$:

- $I = 1$: a warp per output, coalesced lane-strided reads, shuffle
  reduction of $(v, j)$.
- $I > 1$: a thread per output, looping over $R$ with stride $I$, coalesced
  across neighbouring threads.

## Cost analysis

$$
Q = 4ORI + 4OI\ \text{bytes}, \qquad T_{\min} = \frac{4ORI}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes |
| $\beta$ | DRAM bandwidth |

This is a memory-bound single pass.

## Pitfalls

- **Tie-breaking** as for argmax.
- **Negative zero**: $-0.0 = +0.0$ compares equal, so the first index wins,
  as in PyTorch.

## Verification

All test cases pass on [cuemu](../../tools/cuemu/README.md) with exact
index equality.

## Related

- [Argmax](../argmax/), [Min over Dimension](../min-dim/).

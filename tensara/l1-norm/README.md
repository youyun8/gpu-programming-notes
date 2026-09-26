---
title: L1 Normalization
platform: Tensara
upstream: l1-norm
url: https://tensara.org/problems/l1-norm
difficulty: easy
tags: [normalization, reduction, row-per-block]
status: solved
---

# L1 Normalization

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/l1-norm)

## Problem

Row-wise L1 normalization of a $B\times D$ float32 matrix
($B = 128, 256$; $D = 4096 \dots 16384$): divide every row by the sum of
its absolute values plus $\epsilon = 10^{-10}$. The check is
`rtol = 7e-4`, `atol = 5e-5`.

## Formulation

$$
s_b = \sum_{d=0}^{D-1} \lvert x_{bd} \rvert, \qquad y_{bd} = \frac{x_{bd}}{s_b + \epsilon}
$$

| Symbol | Meaning |
|---|---|
| $B, D$ | number of rows, row length |
| $x_{bd}, y_{bd}$ | input and output element in row $b$, column $d$ |
| $s_b$ | L1 norm of row $b$ |
| $\epsilon$ | $10^{-10}$, added to the norm (not clamped) as in the reference |

After normalization $\sum_d \lvert y_{bd}\rvert = s_b/(s_b + \epsilon) \approx 1$.

## Approach

**One block per row**, 256 threads:

1. Each thread strides the row and adds $\lvert x\rvert$; a block
   reduction (warp `__shfl_xor_sync`, then one shared-memory hop) gives
   $s_b$ to every thread.
2. Each thread computes $r = 1/(s_b + \epsilon)$ once and writes
   $y = x\,r$ for its elements. The row (at most 64 KB) was just read, so
   the second pass mostly hits in L2.

## Cost Analysis

$$
Q_{\text{DRAM}} \approx 4BD + 4BD = 8BD\ \text{bytes}, \qquad \#\text{blocks} = B
$$

| Symbol | Meaning |
|---|---|
| $Q_{\text{DRAM}}$ | read once from DRAM (second read from L2), write once |
| #blocks | one per row |

At $B = 256$, $D = 8192$: 16.8 MB, about 8 µs at 2 TB/s. With only 128 to
256 blocks, each SM gets one or two; the latency of the row walk is then
the limit. Splitting rows over a cluster of blocks (distributed shared
memory on Hopper) or several warps per row with more threads would help.

## Pitfalls

- **$\epsilon$ is added**, not used as a lower clamp.
- **Absolute values**: summing signed $x$ is a different (and wrong)
  normalization.
- **Multiply by the reciprocal**: 1 ulp from the reference's division.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [L2 Norm](../l2-norm/), [Frobenius Norm](../frobenius-norm/), [RMS Norm](../rms-norm/).

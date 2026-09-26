---
title: Cumulative Product
platform: Tensara
upstream: cumprod
url: https://tensara.org/problems/cumprod
difficulty: medium
tags: [scan, prefix-product, fp64-accumulation]
status: solved
---

# Cumulative Product

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/cumprod)

## Problem

Inclusive prefix product of a float32 vector of length $N$ (64 K … 1 M),
matching `torch.cumprod(x, dim=0)`. Products of many random numbers
quickly underflow to 0 or overflow to $\infty$, which is why the check is
loose (`rtol = 1e-2`, `atol = 2e-2`); the interesting part is the scan.

## Formulation

$$
y_i = \prod_{j=0}^{i} x_j = y_{i-1}\,x_i, \qquad y_{-1} = 1
$$

| Symbol | Meaning |
|---|---|
| $x_j$ | Input element |
| $y_i$ | Output: product of the first $i+1$ inputs |
| 1 | Identity of multiplication |

A scan works for any **associative** operator $\otimes$ with identity $e$.
Splitting the input into chunks $c$ of length $L$:

$$
T_c = \bigotimes_{j \in c} x_j, \qquad
E_c = \bigotimes_{c' < c} T_{c'}, \qquad
y_i = E_{c(i)} \otimes \bigotimes_{j = cL}^{i} x_j
$$

| Symbol | Meaning |
|---|---|
| $\otimes, e$ | The scan operator and its identity ($\times$ and 1 here) |
| $L$ | Chunk length (2048) |
| $T_c$ | Total of chunk $c$ |
| $E_c$ | Exclusive carry into chunk $c$: the total of all earlier chunks |
| $c(i)$ | The chunk that contains $i$ |

## Approach

The same three-kernel **reduce-then-scan** as [cumsum](../cumsum/), with
the operator as a template parameter (`Times`: identity 1, apply $a\cdot b$):

1. `chunkTotals`: each 256-thread block folds its 2048 elements (8 per
   thread) to $T_c$.
2. `scanTotals`: one block scans the $\lceil N/2048\rceil \le 512$ totals
   into exclusive carries $E_c$.
3. `scanChunks`: each block re-reads its chunk, scans it (per-thread
   serial scan of 8 items, then a warp-shuffle scan and a scan of warp
   totals) and applies $E_c$.

All intermediate values are `double`. For products this matters more than
for sums: fp64's exponent range ($10^{\pm308}$) keeps partial products
exact far longer, and they are rounded to float only at the store.

## Cost Analysis

$$
Q = 4N\ (\text{read}) + 4N\ (\text{re-read}) + 4N\ (\text{write}) = 12N\ \text{bytes}, \qquad \#\text{launches} = 3
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes |
| #launches | Kernel launches per call |

For $N = 2^{20}$, $Q = 12.6$ MB, about 6 µs at 2 TB/s, so launch
overhead and the single-block middle kernel are comparable to the data
movement. A decoupled look-back scan (single pass, $8N$ bytes) is the
state-of-the-art alternative.

## Pitfalls

- **Identity is 1**, not 0: padding lanes past $N$ must not zero the
  product.
- **fp64 multiplications** are slow on consumer GPUs (1/64 rate), but here
  there are only a few per element.
- **0 × ∞**: once a prefix hits $\infty$ and a later one is 0, both
  PyTorch and this code give NaN; the checker compares NaNs as equal.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Cumsum](../cumsum/), [Running Sum 1D](../running-sum-1d/),
  LeetGPU [Prefix Sum](../../leetgpu/016-prefix-sum/).

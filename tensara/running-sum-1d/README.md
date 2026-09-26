---
title: 1D Running Sum
platform: Tensara
upstream: running-sum-1d
url: https://tensara.org/problems/running-sum-1d
difficulty: easy
tags: [scan, prefix-sum, sliding-window, fp64-accumulation]
status: solved
---

# 1D Running Sum

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/running-sum-1d)

## Problem

Sliding-window sum of a float32 signal of length $N$ (32 K … 512 K) with a
window of $W = 8191$ elements, as a `conv1d` with a kernel of ones and
zero padding $\lfloor W/2 \rfloor$. The check is `rtol = 5e-4`,
`atol = 3e-2`.

## Formulation

$$
h = \left\lfloor \frac{W}{2} \right\rfloor, \qquad
\text{out}[i] = \sum_{j=0}^{W-1} \tilde{x}[\,i + j - h\,], \qquad 0 \le i < L = N + 2h - W + 1
$$

| Symbol | Meaning |
|---|---|
| $x$ | Input signal, length $N$; $\tilde{x}$ is $x$ with zeros outside $[0, N)$ |
| $W$ | Window length |
| $h$ | Padding, $\lfloor W/2\rfloor$ |
| $L$ | Output length ($= N$ for odd $W$) |
| $i$ | Output index |

With the inclusive prefix sum $\Pi$, every window is a difference of two
prefix values:

$$
\Pi[t] = \sum_{q=0}^{t} x[q], \qquad
\ell = \max(i - h, 0), \quad r = \min(i - h + W - 1,\ N - 1), \qquad
\text{out}[i] = \Pi[r] - \Pi[\ell - 1]
$$

| Symbol | Meaning |
|---|---|
| $\Pi[t]$ | Inclusive prefix sum, with $\Pi[-1] = 0$ |
| $\ell, r$ | First and last in-bounds input index of window $i$ |

This turns $O(NW)$ additions into $O(N)$.

## Approach

1. **Inclusive scan in fp64** into a temporary `double` array, using the
   three-kernel reduce-then-scan of [Cumsum](../cumsum/) (2048-element
   chunks, fp64 carries), with the output type changed to `double`.
2. **`windowSums`**: one thread per output reads $\Pi[r]$ and
   $\Pi[\ell - 1]$, subtracts in `double`, and rounds once to float.

Why fp64: $\Pi$ grows to $\sim\sqrt{N}$ or more (for random data) while a
window sum is much smaller, and the subtraction cancels the leading
digits. In fp32 the absolute error would be about $u\,\lvert\Pi\rvert$:

$$
\bigl\lvert \Delta\text{out} \bigr\rvert \lesssim 2u\,\max_t \lvert \Pi[t] \rvert, \qquad u_{32} = 2^{-24},\ u_{64} = 2^{-53}
$$

| Symbol | Meaning |
|---|---|
| $\Delta\text{out}$ | Error of one window sum from rounded prefixes |
| $u_{32}, u_{64}$ | Unit roundoff of float and double |

## Cost Analysis

$$
Q \approx 4N + 4N\ (\text{scan reads}) + 8N\ (\text{write }\Pi) + 16N\ (\text{read two }\Pi) + 4N\ (\text{write out}) = 36N\ \text{bytes}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes (the two $\Pi$ reads per output mostly hit L2 since neighbouring outputs read neighbouring prefixes) |

At $N = 512$ K this is ~19 MB, around 10 µs; the direct method would be
$4\times10^9$ additions, milliseconds. The shared-memory approach of
[Conv 1D](../conv-1d/) also works, but the scan exploits that all weights
are 1.

## Pitfalls

- **Output length** $L = N + 2h - W + 1$ (equal to $N$ for odd $W$).
- **Boundary windows** are clipped at both ends ($\ell$ and $r$), which is
  exactly zero padding for a sum.
- **fp32 prefix sums** fail the tolerance for large $N$.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Cumsum](../cumsum/), [Conv 1D](../conv-1d/), [Box Blur](../box-blur/),
  LeetGPU [Subarray Sum](../../leetgpu/047-subarray-sum/).

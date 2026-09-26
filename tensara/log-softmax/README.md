---
title: Log Softmax
platform: Tensara
upstream: log-softmax
url: https://tensara.org/problems/log-softmax
difficulty: easy
tags: [softmax, online-softmax, warp-per-row]
status: solved
---

# Log Softmax

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/log-softmax)

## Problem

Row-wise log-softmax of an $M\times N$ float32 matrix ($4096^2$ …
$8192^2$), matching `F.log_softmax(x, dim=1)`. The check is
`rtol = 1e-4`, `atol = 2e-5`.

## Formulation

$$
y_{ij} = \ln\frac{e^{x_{ij}}}{\sum_{k} e^{x_{ik}}} = x_{ij} - \operatorname{LSE}_i, \qquad
\operatorname{LSE}_i = m_i + \ln\sum_{k=0}^{N-1} e^{x_{ik} - m_i}, \qquad m_i = \max_k x_{ik}
$$

| Symbol | Meaning |
|---|---|
| $x_{ij}, y_{ij}$ | Input and output of row $i$, column $j$ |
| $\operatorname{LSE}_i$ | log-sum-exp of row $i$ |
| $m_i$ | Row maximum; subtracting it keeps every exponent $\le 0$ (no overflow) |

The maximum and the sum are computed together in one pass with the
**online** update of a pair $(m, s)$, where $s = \sum e^{x - m}$:

$$
(m_1, s_1) \oplus (m_2, s_2) = \Bigl(M,\ s_1 e^{m_1 - M} + s_2 e^{m_2 - M}\Bigr), \qquad M = \max(m_1, m_2)
$$

| Symbol | Meaning |
|---|---|
| $(m, s)$ | Running maximum and running sum of $e^{x - m}$ |
| $\oplus$ | Associative merge; a new element $x$ is merged as $(x, 1)$ |
| $M$ | The larger of the two maxima |

## Approach

1. **One warp per row.** Each lane merges its strided elements into a
   private $(m, s)$ pair with $\oplus$; the 32 pairs are combined with a
   5-step `__shfl_xor_sync` butterfly, so every lane ends with the row's
   $(m_i, s_i)$.
2. $\operatorname{LSE}_i = m_i + \ln s_i$.
3. **Write pass**: $y_{ij} = x_{ij} - \operatorname{LSE}_i$ (no exponentials).

## Cost Analysis

$$
Q = 4MN\ (\text{read}) + 4MN\ (\text{re-read}) + 4MN\ (\text{write}), \qquad \#\exp = MN\ (\text{plus merges})
$$

| Symbol | Meaning |
|---|---|
| $Q$ | Bytes; the re-read hits L2 when a row (up to 32 KB) is still resident |
| #exp | Exponentials in the online pass |

At $8192^2$: 268 MB in and out, ~0.27 ms at 2 TB/s. Unlike softmax, the
write pass needs no second exponential.

## Pitfalls

- **Stability**: $\ln\sum e^{x}$ overflows for $x > 88$; always subtract the
  maximum.
- **Initial $m = -\text{FLT\_MAX}$** (not $-\infty$), so that
  $e^{m_1 - M}$ never evaluates $e^{-\infty + \infty}$.
- **Do not compute $\ln(\operatorname{softmax})$**: small probabilities
  underflow to 0 and give $-\infty$.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Softmax](../softmax/), [KL Loss](../kl-loss/),
  LeetGPU [Softmax](../../leetgpu/005-softmax/),
  LeetGPU [Categorical Cross-Entropy](../../leetgpu/025-categorical-cross-entropy-loss/).

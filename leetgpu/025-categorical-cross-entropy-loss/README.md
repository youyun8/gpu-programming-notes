---
title: Categorical Cross Entropy Loss
platform: LeetGPU
upstream: medium/25_categorical_cross_entropy_loss
url: https://leetgpu.com/challenges/categorical-cross-entropy-loss
difficulty: medium
tags: [reduction, logsumexp, warp-per-row, loss]
status: solved
---

# Categorical Cross Entropy Loss

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/categorical-cross-entropy-loss)

## Problem

Mean categorical cross-entropy over a batch: `logits` is an $N \times C$
float32 matrix and `true_labels` holds $N$ class indices
($1 \le N \le 10^4$, $2 \le C \le 1000$, $\lvert z\rvert \le 10$; tolerance
`1e-5`). The output is a single float. This is the standard classification
loss. The numerically important piece is the **log-sum-exp**, which must
never be computed as `log(sum(exp(z)))` naively.

## Formulation

$$
\mathcal L = \frac{1}{N}\sum_{j=0}^{N-1} \ell_j, \qquad
\ell_j = -\log \frac{e^{z_{j,y_j}}}{\sum_{k=0}^{C-1} e^{z_{jk}}} = \operatorname{LSE}(\mathbf z_j) - z_{j, y_j}
$$

$$
\operatorname{LSE}(\mathbf z_j) = \log\sum_{k=0}^{C-1} e^{z_{jk}} = m_j + \log\sum_{k=0}^{C-1} e^{z_{jk} - m_j}, \qquad m_j = \max_k z_{jk}
$$

| Symbol | Meaning |
|---|---|
| $N$ | batch size (number of samples) |
| $C$ | number of classes |
| $z_{jk}$ | logit of sample $j$ for class $k$ (row-major, offset $jC + k$) |
| $\mathbf z_j$ | row $j$ of the logits |
| $y_j$ | true class of sample $j$, $0 \le y_j < C$ |
| $\ell_j$ | loss of sample $j$ (negative log-likelihood of the true class) |
| $\operatorname{LSE}$ | log-sum-exp; the max shift $m_j$ keeps every exponent $\le 0$ |
| $\mathcal L$ | batch mean loss, written to `loss[0]` |

The LSE is computed in one pass with the online $(m, s)$ pair merge from
[Softmax](../005-softmax/):
$(m_1,s_1)\oplus(m_2,s_2) = (m, s_1e^{m_1-m} + s_2e^{m_2-m})$, with
$\operatorname{LSE} = m + \log s$.

## Approach

1. **`sampleLosses`** (≤ 1024 blocks × 8 warps). One **warp per sample**
   (grid-stride over rows):
   - Lane $\ell$ streams the logits $\ell, \ell+32, \dots$ of the row
     (coalesced) and folds each into its $(m, s)$ pair.
   - 5 `__shfl_xor_sync` butterfly steps merge the 32 pairs. After an XOR
     butterfly, *every* lane holds the full result.
   - Lane 0 computes $\ell_j = m + \log s - z_{j,y_j}$ and adds it to a
     float64 running total.
   - The warp totals of a block are combined through shared memory into one
     float64 partial per block.
2. **`finalMean`** (1 block). Sums the partials in float64 and divides by $N$.

Warp-per-row fits $C \le 1000$: a row is at most 32 coalesced loads per
lane-group, and no shared memory is needed for the row reduction.

## Cost Analysis

$$
Q \approx 4NC + 4N, \qquad W \approx NC\,(\text{1 exp} + 3\ \text{flops}), \qquad T_{\min} \approx \frac{4NC}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | bytes: every logit once, plus the labels |
| $W$ | work, dominated by one `expf` per logit (the online merge also rescales $s$) |
| $\beta$ | DRAM bandwidth |

$N = 10^4$, $C = 1000$: 40 MB of logits, so ≈ 20 µs of bandwidth. The
per-logit `expf` (≈ 20 instructions each) puts it near the balance point.
It is still roughly memory-bound on large GPUs.

## Pitfalls

- **Naive log-sum-exp.** $e^{10}$ is harmless here, but the same code
  overflows for logits > 88. Always shift by the max.
- **Precision of the mean.** Adding $10^4$ losses of magnitude ~7 in float32
  loses ~4 digits. The float64 per-block and final sums avoid that.
- **Label indexing.** $z_{j,y_j}$ is read by lane 0 only, once per row.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-5`,
including $C = 2$ and $C = 1000$.

## Related

- [Softmax](../005-softmax/), Tensara [Log-Softmax](../../tensara/log-softmax/),
  [KL Divergence](../../tensara/kl-loss/), [DPO Loss](../108-dpo-sequence-loss/).

---
title: Sliding Window Self-Attention
platform: LeetGPU
upstream: hard/59_sliding_window_attn
url: https://leetgpu.com/challenges/sliding-window-self-attention
difficulty: hard
tags: [attention, sliding-window, flash-attention, local-attention]
status: solved
---

# Sliding Window Self-Attention

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/sliding-window-self-attention)

## Problem

Sliding-window self-attention: query $i$ attends only to keys $j$ with
$\lvert i - j\rvert \le w$ (a symmetric, non-causal band), on
$Q, K, V \in \mathbb R^{M\times d}$ (tolerance `1e-5`). This is the local
attention pattern of Longformer and Mistral (Mistral uses a causal variant).
Cost drops from $O(M^2 d)$ to $O(M w d)$ **if the kernel never touches keys
outside the band**.

## Formulation

$$
s_{ij} = \frac{\mathbf q_i\cdot\mathbf k_j}{\sqrt d}, \qquad
\mathcal W_i = \{\, j : \max(0, i - w) \le j \le \min(M-1, i + w) \,\}, \qquad
O_{i,:} = \sum_{j\in\mathcal W_i} \frac{e^{s_{ij} - m_i}}{\sum_{j'\in\mathcal W_i} e^{s_{ij'} - m_i}}\,\mathbf v_j
$$

| Symbol | Meaning |
|---|---|
| $M$ | Sequence length |
| $d$ | Head dimension |
| $w$ | Window radius (`window_size`) |
| $\mathcal W_i$ | Visible keys of query $i$: at most $2w + 1$ of them |
| $s_{ij}$ | Scaled score |
| $m_i$ | Max over the window |
| $O_{i,:}$ | Output row |

## Approach

The FlashAttention-style kernel (warp per query row, 8 rows per block,
32-key shared tiles, lane-per-key scoring, online softmax) with two
band-specific changes:

1. **Block key range.** A block covering rows $[r_0, r_1]$ only needs keys
   $[\max(0, r_0 - w),\ \min(M-1, r_1 + w)]$. The tile loop runs over exactly
   this range, $\le 8 + 2w$ keys instead of $M$.
2. **Per-lane mask.** Inside the range, lane $\ell$'s key $j$ is valid for
   the warp's row $i$ only if $\lvert i - j\rvert \le w$. Otherwise its score
   is $-\infty$ and its weight 0.

The tile loop bounds are block-uniform, so every warp reaches every
`__syncthreads()`.

## Cost Analysis

$$
W \approx 4d\sum_{i}\lvert\mathcal W_i\rvert \le 4dM(2w+1), \qquad \text{tiles per block} = \left\lceil\frac{(r_1 - r_0 + 1) + 2w}{32}\right\rceil
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs for scores and $PV$ over visible pairs |
| $r_0,\ r_1$ | First and last query row of the block |

Linear in $M$ for fixed $w$. For small $w$, most lanes in a tile are masked
(a tile has 32 keys, the band per row $2w+1$), so efficiency improves with
larger $w$ or with more rows per block.

## Pitfalls

- **Symmetric window.** The reference masks $\lvert j - i\rvert > w$ on both
  sides; this is not causal.
- **Clamping at the ends.** Rows near 0 or $M-1$ see fewer keys, and the
  denominators differ per row.
- **Barrier uniformity**, as for [Causal Attention](../053-casual-attention/).

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-5`,
including $w = 0$ (each query sees only itself: output = $V$), and $w \ge M$
(full attention).

## Related

- [Causal Attention](../053-casual-attention/), [Softmax Attention](../006-softmax-attention/),
  [Attention with Sinks](../112-attention-with-sinks/).

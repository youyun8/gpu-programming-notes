---
title: Scaled Dot-Product Attention
platform: Tensara
upstream: scaled-dot-attention
url: https://tensara.org/problems/scaled-dot-attention
difficulty: hard
tags: [attention, flash-attention, online-softmax, shared-memory]
status: solved
---

# Scaled Dot-Product Attention

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/scaled-dot-attention)

## Problem

Non-causal scaled dot-product attention for tensors $Q, K, V$ of shape
$(B, H, S, E)$, matching `F.scaled_dot_product_attention` without mask or
dropout. Test shapes range from $(16, 32, 256, 64)$ to $(8, 16, 2048, 64)$
and $(8, 16, 512, 256)$. The check is `rtol = 2e-2`, `atol = 5e-3`.

## Formulation

For every batch $b$ and head $h$ independently:

$$
s_{ij} = \frac{\mathbf{q}_i\cdot\mathbf{k}_j}{\sqrt{E}}, \qquad
P_{ij} = \frac{e^{s_{ij} - m_i}}{\sum_{j'} e^{s_{ij'} - m_i}}, \qquad
\mathbf{o}_i = \sum_{j=0}^{S-1} P_{ij}\,\mathbf{v}_j, \qquad m_i = \max_j s_{ij}
$$

| Symbol | Meaning |
|---|---|
| $B, H, S, E$ | batch, heads, sequence length, head dimension |
| $\mathbf{q}_i, \mathbf{k}_j, \mathbf{v}_j$ | row $i$ of $Q$, rows $j$ of $K$ and $V$ (for one $(b, h)$) |
| $s_{ij}$ | scaled score |
| $P_{ij}$ | attention probability (softmax over $j$) |
| $m_i$ | row maximum of the scores |
| $\mathbf{o}_i$ | output row, length $E$ |

The **online softmax** processes keys in tiles $\mathcal{J}_t$ of 32 and
keeps a running max $m$, normaliser $\ell$ and unnormalised output
$\mathbf{u}$:

$$
m' = \max\bigl(m, \max_{j\in\mathcal{J}_t} s_{ij}\bigr), \quad
\ell' = \ell\,e^{m - m'} + \sum_{j\in\mathcal{J}_t} e^{s_{ij} - m'}, \quad
\mathbf{u}' = \mathbf{u}\,e^{m - m'} + \sum_{j\in\mathcal{J}_t} e^{s_{ij} - m'}\mathbf{v}_j
$$

and at the end $\mathbf{o}_i = \mathbf{u}/\ell$.

| Symbol | Meaning |
|---|---|
| $\mathcal{J}_t$ | the $t$-th tile of 32 keys |
| $m, \ell, \mathbf{u}$ | running maximum, running sum of exponentials, running weighted sum of values |
| $m', \ell', \mathbf{u}'$ | their values after tile $t$ |
| $e^{m - m'}$ | rescaling factor applied when the maximum grows |

## Approach

A FlashAttention-style fused kernel (`flashForward`), with the $B\cdot H$
pairs folded into the grid (head stride $S\cdot E$):

1. **4 warps per block, one query row per warp.**
2. **Scoring**: for each tile of 32 keys, lane $l$ computes the score of
   key $l$ (a length-$E$ dot product with the query row).
3. **Streaming K and V** through shared memory in 128-wide slices of the
   head dimension, so shared memory does not grow with $E$ (up to 1024).
4. **Online softmax** per warp: warp max and sum of the 32 scores, rescale
   the accumulator.
5. **Accumulation**: lane $l$ owns output columns $l, l+32, \dots$ and adds
   $\sum_j p_j v_{j,c}$; the probability of key $j$ is broadcast from lane
   $j$ with a shuffle.

The $S\times S$ score matrix is never materialised: memory is $O(SE)$ per
head instead of $O(S^2)$.

## Cost Analysis

$$
W = 4\,BHS^2E\ \text{flops}, \qquad Q_{\min} = 4\cdot 4\,BHSE\ \text{bytes}, \qquad
Q_{\text{K,V}} \approx \frac{S}{4}\cdot 8\,BHSE\ \text{bytes through L2}
$$

| Symbol | Meaning |
|---|---|
| $W$ | two matrix products ($QK^{\mathsf T}$ and $PV$), 2 flops per FMA |
| $Q_{\min}$ | compulsory DRAM bytes: read $Q, K, V$, write $O$ |
| $Q_{\text{K,V}}$ | K/V traffic: every block (4 query rows) re-streams the head's $K$ and $V$ |

At $(8, 16, 2048, 64)$: $W = 275$ GFLOP. Because each block owns only 4
query rows, $K$ and $V$ are re-read $S/4$ times through L2; larger query
tiles (64–128 rows per block, as in FlashAttention-2) and tensor-core MMA
would cut both the L2 traffic and the instruction count.

## Pitfalls

- **Scale** $1/\sqrt{E}$ applied to the scores, not to $V$ or the output.
- **Stability**: subtract the running maximum; initialise $m$ to a large
  negative finite value to avoid $\infty - \infty$.
- **No mask**: this is bidirectional attention.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Softmax](../softmax/), LeetGPU [Softmax Attention](../../leetgpu/006-softmax-attention/),
  LeetGPU [Multi-Head Attention](../../leetgpu/012-multi-head-attention/),
  LeetGPU [Causal Attention](../../leetgpu/053-casual-attention/).

---
title: Softmax Attention Backward
platform: LeetGPU
upstream: medium/111_softmax_attention_backward
url: https://leetgpu.com/challenges/softmax-attention-backward
difficulty: medium
tags: [attention, backward, flash-attention, autograd]
status: solved
---

# Softmax Attention Backward

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/softmax-attention-backward)

## Problem

The **backward pass** of single-head softmax attention. Given $Q$ ($M\times d$),
$K, V$ ($N\times d$) and the upstream gradient $dO$ ($M\times d$), compute
$dQ$, $dK$ and $dV$ ($M, N \le 10^5$, $d \le 128$; benchmark $M = 8192$,
$N = 4096$, $d = 128$; tolerance `1e-4`). Training transformers needs this
kernel as much as the forward. Like the forward pass, it must avoid storing
the $M\times N$ probability matrix: 128 MB at the benchmark size, 40 GB at the
upper limit.

## Formulation

Forward: $S = QK^{\mathsf T}/\sqrt d$, $P = \operatorname{softmax}_{\text{row}}(S)$, $O = PV$. Backward:

$$
dV = P^{\mathsf T}dO, \qquad dP = dO\,V^{\mathsf T}, \qquad
dS_{ij} = P_{ij}\bigl(dP_{ij} - D_i\bigr),\ \ D_i = \sum_k P_{ik}\,dP_{ik}, \qquad
dQ = \frac{dS\,K}{\sqrt d}, \qquad dK = \frac{dS^{\mathsf T}Q}{\sqrt d}
$$

$$
D_i = \sum_k P_{ik}\,(dO_i\cdot V_k) = dO_i\cdot\Bigl(\sum_k P_{ik}V_k\Bigr) = dO_i\cdot O_i, \qquad
P_{ij} = e^{S_{ij} - L_i},\ \ L_i = \log\sum_j e^{S_{ij}}
$$

| Symbol | Meaning |
|---|---|
| $M,\ N,\ d$ | Queries, keys, head dimension |
| $S,\ P$ | Scaled scores and attention probabilities ($M\times N$, never stored) |
| $O$ | Forward output |
| $dO$ | Upstream gradient of the loss w.r.t. $O$ |
| $dP$ | Gradient w.r.t. $P$: $dP_{ij} = dO_i\cdot V_j$ |
| $dS$ | Gradient w.r.t. the scores (softmax Jacobian applied row-wise) |
| $D_i$ | Row correction term; equal to $dO_i\cdot O_i$ |
| $L_i$ | Row log-sum-exp; lets $P_{ij}$ be recomputed from $S_{ij}$ alone |
| $dQ,\ dK,\ dV$ | Outputs |

**The softmax Jacobian.** For $\mathbf p = \operatorname{softmax}(\mathbf s)$,
$\partial p_j/\partial s_k = p_j(\delta_{jk} - p_k)$. So
$ds_k = \sum_j dp_j\,p_j(\delta_{jk} - p_k) = p_k(dp_k - \sum_j p_j dp_j)$,
which is the $dS$ formula above.

## Approach

The FlashAttention-2 backward strategy: store only two scalars per query row,
and recompute $P$ on the fly.

1. **`rowStats`** (query-parallel): rerun the forward with the online softmax
   (warp per query row, 32-key tiles). Store $L_i = m + \log\ell$ and
   $D_i = dO_i\cdot O_i$, with $O_i$ taken from the accumulators.
2. **`gradQ`** (query-parallel): for each key tile, lane $\ell$ computes
   $S_{ij}$ and $dP_{ij}$ for key $j = j_0 + \ell$, then
   $dS_{ij} = e^{S_{ij} - L_i}(dP_{ij} - D_i)/\sqrt d$. Each $dS_{ij}$ is
   shuffled to all lanes, which accumulate $dQ_i \mathrel{+}= dS_{ij}K_j$
   into their $d/32$ columns.
3. **`gradKV`** (key-parallel): a warp owns key row $j$ and streams tiles of
   **queries** (with their $L_i$, $D_i$). Lane $\ell$ evaluates the pair
   with query $i_0 + \ell$, and the lanes accumulate
   $dV_j \mathrel{+}= P_{ij}\,dO_i$ and $dK_j \mathrel{+}= dS_{ij}\,Q_i$.

**No atomics.** $dQ$ rows depend on all keys, and $dK$/$dV$ rows depend on all
queries. Using a query-parallel pass for $dQ$ and a key-parallel pass for
$dK$/$dV$ gives every output row exactly one owner warp. The price is
computing $S$ and $dP$ twice, a standard trade-off (FlashAttention-2 instead
uses atomics for $dQ$ in a single pass).

## Cost Analysis

$$
W \approx \underbrace{4MNd}_{\text{rowStats}} + \underbrace{6MNd}_{\text{gradQ}} + \underbrace{8MNd}_{\text{gradKV}} = 18MNd, \qquad
\text{extra memory} = 8M\ \text{bytes}\ (L, D)
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs: every pass recomputes $QK^{\mathsf T}$ (and $dO V^{\mathsf T}$ where needed) for all pairs |
| Extra memory | The only intermediate state: two floats per query row |

Benchmark: $W \approx 7.7\times10^{10}$ FLOP. This is compute-bound and about
2.5× the forward pass, matching the usual "backward ≈ 2–3× forward" rule of
thumb.

## Pitfalls

- **Scale placement.** $S$ uses $1/\sqrt d$ (pre-applied to $Q$ in the
  shared buffer, or multiplied in `gradKV`), and both $dQ$ and $dK$ carry
  another $1/\sqrt d$. The two factors are easy to confuse.
- **$D_i$ from $O_i$.** Computing $\sum_k P_{ik}dP_{ik}$ directly would need
  a full pass over the keys. $dO_i\cdot O_i$ is the same number.
- **Masked lanes** past $N$ (or $M$) must contribute zero weights, not NaN.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-4`,
and the gradients were cross-checked against `torch.autograd` on random inputs.

## Related

- [Softmax Attention](../006-softmax-attention/) (forward), [Multi-Head Attention](../012-multi-head-attention/).

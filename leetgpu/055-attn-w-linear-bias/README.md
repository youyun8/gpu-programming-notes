---
title: Attention with Linear Biases
platform: LeetGPU
upstream: medium/55_attn_w_linear_bias
url: https://leetgpu.com/challenges/attention-with-linear-biases
difficulty: medium
tags: [attention, alibi, gemm, softmax, fused-epilogue]
status: solved
---

# Attention with Linear Biases

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/attention-with-linear-biases)

## Problem

Attention with Linear Biases (**ALiBi**, Press et al. 2022). Instead of
positional embeddings, a bias proportional to the query–key distance is
added to the scores: $Q \in \mathbb R^{M \times d}$,
$K, V \in \mathbb R^{N\times d}$, slope $\alpha \in [-1, 1]$
($M, N \le 2048$, $d \le 1024$; benchmark $M = N = 2048$; tolerance `1e-4`).

## Formulation

$$
S_{ij} = \frac{\mathbf q_i\cdot\mathbf k_j}{\sqrt d} + \alpha\,(i - j), \qquad
P_{ij} = \frac{e^{S_{ij} - m_i}}{\sum_{j'} e^{S_{ij'} - m_i}}, \qquad
O = PV
$$

| Symbol | Meaning |
|---|---|
| $M,\ N$ | numbers of queries and keys |
| $d$ | head dimension (up to 1024) |
| $\mathbf q_i,\ \mathbf k_j$ | rows of $Q$ and $K$ |
| $\alpha$ | ALiBi slope (one head, so one slope) |
| $i - j$ | signed relative position (query index minus key index) |
| $S_{ij}$ | biased, scaled score |
| $m_i$ | row maximum of $S$ |
| $P_{ij}$ | attention weight (row softmax) |
| $O$ | output, $M \times d$ |

With $\alpha < 0$ (the usual choice in causal LMs), distant keys are
penalised linearly. This lets models extrapolate to longer sequences than
they were trained on.

## Approach

With $d$ up to 1024, the per-warp flash design used elsewhere would need
large register accumulators. Since $M N \le 4.2$M, the score matrix
(≤ 16 MB) fits comfortably, so a clean **three-kernel** pipeline is used:

1. **$S = QK^{\mathsf T}\cdot\tfrac{1}{\sqrt d} + \alpha(i - j)$** with a
   64 × 64 register-blocked SGEMM in "NT" form (the $B$ operand is read
   transposed). The $B$ tile is loaded with $k$ fastest, so reads of $K$'s
   rows are coalesced. The scale and the ALiBi term are applied in the
   **epilogue**, where each thread already knows its $(i, j)$. The bias costs
   nothing extra.
2. **Row softmax in place**, one warp per row: max, exponentiate and sum,
   normalise.
3. **$O = PV$** with the same SGEMM template in "NN" form.

The template `sgemm<kTransB, kAlibi>` generates both GEMMs from one source
with compile-time branches.

## Cost Analysis

$$
W = 2MNd + 2MNd + O(MN), \qquad Q \approx 4\,(MN\cdot 3) + 4\left(Md + Nd\right)\cdot\frac{\max(M,N)}{64} + 4Md
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs: two GEMMs plus the softmax |
| $Q$ | bytes: write $S$, read and write it in softmax, read it in the second GEMM, plus tiled GEMM operand traffic |

At $M = N = 2048$, $d = 1024$: $W \approx 17$ GFLOP (GEMM-dominated) and
48 MB of score traffic. Materialising $S$ costs about 3 × 16 MB of extra
traffic compared with a fused flash kernel, which is small next to the GEMM
time here.

## Pitfalls

- **Sign of the bias.** It is $\alpha\,(i - j)$ with $i$ the query row and
  $j$ the key column, exactly as the reference's
  `arange(M)[:,None] - arange(N)[None,:]`.
- **Bias before the max.** The bias must be added before the softmax
  max-subtraction (it is part of the score).
- **No causal mask** in this problem. ALiBi alone is not a mask.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-4`,
including $\alpha = \pm1$, $d = 1024$ and $M \ne N$.

## Related

- [Softmax Attention](../006-softmax-attention/), [Causal Attention](../053-casual-attention/),
  [Decaying Causal Attention](../092-decaying-causal-attention/).

---
title: Token Embedding Layer
platform: LeetGPU
upstream: medium/106_token_embedding_layer
url: https://leetgpu.com/challenges/token-embedding-layer
difficulty: medium
tags: [embedding, layernorm, gather, fusion]
status: solved
---

# Token Embedding Layer

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/token-embedding-layer)

## Problem

The input layer of BERT-style models: for each of $B\times T$ tokens, gather
its token embedding and its position embedding, add them, and apply
LayerNorm with learnable $\gamma, \beta$ ($B \le 64$, $T \le 1024$,
$V \le 50\,000$, $P \le 4096$, $D \le 1024$; benchmark $B = 32$, $T = 512$,
$D = 768$; tolerance `1e-4`).

## Formulation

$$
\mathbf s_{b,t} = E_T[\tau_{b,t}] + E_P[\pi_t] \in \mathbb R^{D}, \qquad
\mu_{b,t} = \frac1D\sum_{d} s_{b,t,d}, \qquad
\sigma^2_{b,t} = \frac1D\sum_d\bigl(s_{b,t,d} - \mu_{b,t}\bigr)^2
$$

$$
y_{b,t,d} = \gamma_d\,\frac{s_{b,t,d} - \mu_{b,t}}{\sqrt{\sigma^2_{b,t} + \varepsilon}} + \beta_d
$$

| Symbol | Meaning |
|---|---|
| $B,\ T$ | batch and sequence length |
| $V,\ P$ | vocabulary size and number of positions |
| $D$ | embedding width |
| $E_T \in \mathbb R^{V\times D}$ | token embedding table |
| $E_P \in \mathbb R^{P\times D}$ | position embedding table |
| $\tau_{b,t}$ | token id of token $(b, t)$ |
| $\pi_t$ | position id of time step $t$ (shared by all batch rows) |
| $\mathbf s_{b,t}$ | summed embedding |
| $\mu,\ \sigma^2$ | row mean and biased variance |
| $\gamma_d,\ \beta_d$ | LayerNorm scale and shift |
| $y_{b,t,d}$ | output, shape $(B, T, D)$ |

## Approach

**One warp per token**, with the whole row in registers:

1. Read $\tau_{b,t}$ and $\pi_t$. Lane $\ell$ gathers elements
   $\ell, \ell+32, \dots$ of both embedding rows. Every warp access is a
   contiguous 128-byte segment of a table row, a *gather* at row granularity
   but coalesced within the row.
2. Keep $s$ in `vals[32]` (32 × 32 = 1024 ≥ $D$). The loop has a
   compile-time bound, so the array stays in registers.
3. **Two-pass statistics from registers**: sum → shuffle reduction → $\mu$.
   Then $\sum (s - \mu)^2$ → shuffle reduction → $\sigma^2$. Centring before
   squaring avoids the cancellation of $E[s^2] - \mu^2$, and the second pass
   costs no memory traffic because the values are already in registers.
4. Write $\gamma_d (s - \mu)\,\text{rstd} + \beta_d$.

The summed embedding is never written to memory: one fused kernel instead of
gather + add + LayerNorm.

## Cost Analysis

$$
Q \approx \underbrace{8BTD}_{\text{two gathered rows}} + \underbrace{4BTD}_{\text{output}} + \underbrace{8D}_{\gamma,\beta} \ \text{bytes}, \qquad W \approx 8BTD
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes (the position table has only $T$ distinct rows, which hit L2 after the first batch row) |
| $W$ | FLOPs (add, two reductions, normalise, affine) |

Benchmark: $B T D = 12.6$M elements, so ≈ 100–150 MB and ≈ 60 µs. The kernel
is memory-bound.

## Pitfalls

- **Position ids are shared** across the batch: index `position_ids[t]`,
  not `[b, t]`.
- **Unbiased variance** (dividing by $D-1$) would fail the tolerance.
- **Registers for $D = 1024$**: 32 floats per lane plus temporaries is fine.
  For $D \gg 1024$, use a block per row.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-4`,
including $D < 32$ and $D = 1024$.

## Related

- [Layer Normalization](../113-layer-normalization/), [GPT-2 Block](../074-gpt2-block/),
  [ViT Patch Embedding](../118-vit-patch-embedding/).

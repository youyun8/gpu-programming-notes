---
title: Llama Transformer Block
platform: LeetGPU
upstream: hard/93_llama_transformer_block
url: https://leetgpu.com/challenges/llama-transformer-block
difficulty: hard
tags: [transformer, llama, gqa, rope, swiglu, rmsnorm, fusion]
status: solved
---

# Llama Transformer Block

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/llama-transformer-block)

## Problem

One **LLaMA-style decoder block** in float32 with $d = 512$, 8 query heads and
2 KV heads of width 64 (GQA), RoPE, causal attention, and a SwiGLU MLP with
hidden width 1408. None of the projections have biases. Inputs are $x$
($S\times512$), a packed weight buffer, and precomputed RoPE $\cos$/$\sin$
tables ($S \times 32$); tolerance `1e-3`. Compared with the
[GPT-2 block](../074-gpt2-block/), every component is the "modern" variant.

## Formulation

$$
\begin{aligned}
X_1 &= \operatorname{RMSNorm}(X;\ \mathbf w_1), & Q &= X_1W_Q^{\mathsf T},\ \ K = X_1W_K^{\mathsf T},\ \ V = X_1W_V^{\mathsf T} \\
\tilde Q &= \operatorname{RoPE}(Q),\ \ \tilde K = \operatorname{RoPE}(K), & A_h &= \operatorname{softmax}\!\Bigl(\operatorname{mask}_{\text{causal}}\tfrac{\tilde Q_h\tilde K_{\lfloor h/4\rfloor}^{\mathsf T}}{8}\Bigr)V_{\lfloor h/4\rfloor} \\
X' &= X + \operatorname{Concat}(A_0..A_7)\,W_O^{\mathsf T}, & Y &= X' + \Bigl(\operatorname{SiLU}(X_2W_g^{\mathsf T})\odot X_2W_u^{\mathsf T}\Bigr)W_{\text{down}}^{\mathsf T},\ \ X_2 = \operatorname{RMSNorm}(X';\ \mathbf w_2)
\end{aligned}
$$

$$
\operatorname{RMSNorm}(\mathbf z; \mathbf w) = \frac{\mathbf z}{\sqrt{\frac1d\sum_i z_i^2 + 10^{-5}}}\odot\mathbf w, \qquad
\operatorname{RoPE}([\mathbf q_1\,|\,\mathbf q_2]) = [\mathbf q_1\odot\mathbf c - \mathbf q_2\odot\mathbf s\ \,|\ \,\mathbf q_1\odot\mathbf s + \mathbf q_2\odot\mathbf c]
$$

| Symbol | Meaning |
|---|---|
| $S$ | Sequence length |
| $d$ | Model width, 512 |
| $X$ | Input, $S\times d$ |
| $\mathbf w_1,\ \mathbf w_2$ | RMSNorm weights (length $d$) |
| $W_Q$ | $512\times512$ (8 heads × 64), `nn.Linear` layout (out, in) |
| $W_K,\ W_V$ | $128\times512$ each (2 KV heads × 64) |
| $\tilde Q_h,\ \tilde K_g$ | RoPE-rotated query head $h$ and key head $g$ |
| $\lfloor h/4\rfloor$ | GQA mapping: 4 query heads share each KV head |
| 8 | $\sqrt{64}$, the attention scale divisor |
| $\operatorname{mask}_{\text{causal}}$ | $-\infty$ above the diagonal |
| $\mathbf c,\ \mathbf s$ | RoPE $\cos$/$\sin$ row of the token (length 32, shared by both halves) |
| $\mathbf q_1,\ \mathbf q_2$ | First and second 32-element halves of a head vector |
| $W_O$ | Output projection $512\times512$ |
| $W_g,\ W_u$ | Gate and up projections $1408\times512$ |
| $W_{\text{down}}$ | Down projection $512\times1408$ |
| $Y$ | Block output |

## Approach

| # | Kernel | Output | Notes |
|---|---|---|---|
| 1 | `rmsNormRows` | $X_1$ | Warp per row |
| 2 | NT-GEMM, 768 cols | $[Q \mid K \mid V]$ | $W_Q, W_K, W_V$ are **contiguous** in the buffer, so one GEMM computes all three |
| 3 | `applyRope` | Rotate $Q$ and $K$ heads in place | Pair $(j, j+32)$ per thread |
| 4 | Flash attention | $A$ | Causal, GQA `group = 4`, strided reads from the packed `qkv` rows |
| 5 | NT-GEMM + `ResidualEpi` | $X' = X + AW_O^{\mathsf T}$ | Residual fused |
| 6 | `rmsNormRows` | $X_2$ | – |
| 7 | NT-GEMM, 2816 cols | $[G \mid U]$ | $W_g$ and $W_u$ contiguous, so one GEMM |
| 8 | `swiglu` | $\operatorname{SiLU}(G)\odot U$ | Elementwise |
| 9 | NT-GEMM + `ResidualEpi` | $Y = X' + HW_{\text{down}}^{\mathsf T}$ | Residual fused |

**Concatenated projections.** Stacking weight matrices that share the same
input into one taller matrix turns 3 (or 2) skinny GEMMs into one larger GEMM.
The input tile is loaded once, and more blocks run in parallel. This is how
every production LLM implementation handles QKV and gate/up.

**Attention from packed rows.** Head $h$ of $Q$ starts at column $64h$ of a
768-wide row, head $g$ of $K$ at $512 + 64g$, and of $V$ at $640 + 64g$. The
attention kernel receives these offsets and the stride 768, so there are no
reshape or transpose kernels (see [GQA](../080-grouped-query-attention/)).

## Cost Analysis

$$
W \approx 2S d\,(512 + 256) + 2Sd^2 + 2Sd\,(2\cdot1408) + 2S\cdot1408\,d + 2\cdot 8\cdot 64\,S^2
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs: QKV projection, output projection, gate/up, down, and causal attention (≈ half of the dense $4S^2 \cdot 512$) |

For $S = 2048$: the projections take ≈ 9.8 GFLOP and attention ≈ 4.3 GFLOP.
GEMM efficiency again dominates.

## Pitfalls

- **Weight layout.** All projections are `nn.Linear`-style
  $(\text{out}, \text{in})$, i.e. $XW^{\mathsf T}$, hence the NT GEMMs (the
  opposite of the GPT-2 problem).
- **RoPE tables** have 32 columns, reused for both halves of each 64-wide head.
- **GQA mapping** $h \mapsto \lfloor h/4 \rfloor$ (consecutive query heads share).
- **No biases anywhere**, and RMSNorm has no $\beta$.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-3`.

## Related

- [GPT-2 Block](../074-gpt2-block/), [GQA](../080-grouped-query-attention/), [RoPE](../061-rope-embedding/),
  [SwiGLU MLP](../084-swiglu-mlp-block/), [Fused Residual + RMSNorm](../083-fused-residual-add-rms-norm/).

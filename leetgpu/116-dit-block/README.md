---
title: Diffusion Transformer Block
platform: LeetGPU
upstream: hard/116_dit_block
url: https://leetgpu.com/challenges/diffusion-transformer-block
difficulty: hard
tags: [transformer, diffusion, adaln, fusion, attention]
status: solved
cuemu_max_elements: 16777216
---

# Diffusion Transformer Block

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/diffusion-transformer-block)

## Problem

One **DiT block** (Diffusion Transformer: DiT, Stable Diffusion 3, Flux) on a
batch of patch-token sequences $x \in \mathbb R^{B\times S\times 512}$,
conditioned on a per-sample vector $c \in \mathbb R^{B\times512}$ (timestep +
class/text embedding). A packed weight buffer holds the adaLN, QKV, output
and MLP weights (tolerance `1e-3`). Unlike an LLM block, the normalisation
has **no learned affine**. Its scale, shift and a residual **gate** are
predicted per sample from $c$ (adaLN-Zero), so each sample in the batch is
normalised differently.

## Formulation

Modulation (once per sample):

$$
\bigl[\boldsymbol\beta_1\,|\,\boldsymbol\gamma_1\,|\,\mathbf g_1\,|\,\boldsymbol\beta_2\,|\,\boldsymbol\gamma_2\,|\,\mathbf g_2\bigr] = \operatorname{SiLU}(\mathbf c)\,W_{\text{ada}}^{\mathsf T} + \mathbf b_{\text{ada}} \in \mathbb R^{6\cdot512}
$$

Block (per sample, broadcasting the six vectors over all $S$ tokens):

$$
\begin{aligned}
H &= \operatorname{LN}(X)\odot(1 + \boldsymbol\gamma_1) + \boldsymbol\beta_1, &
X' &= X + \mathbf g_1\odot\bigl(\operatorname{MHA}(H)\,W_o^{\mathsf T} + \mathbf b_o\bigr) \\
H' &= \operatorname{LN}(X')\odot(1 + \boldsymbol\gamma_2) + \boldsymbol\beta_2, &
Y &= X' + \mathbf g_2\odot\Bigl(\operatorname{GELU}_{\tanh}\bigl(H'W_1^{\mathsf T} + \mathbf b_1\bigr)W_2^{\mathsf T} + \mathbf b_2\Bigr)
\end{aligned}
$$

| Symbol | Meaning |
|---|---|
| $B,\ S$ | batch size and tokens per sample |
| $X$ | input tokens of one sample, $S\times512$ |
| $\mathbf c$ | conditioning vector of the sample |
| $W_{\text{ada}},\ \mathbf b_{\text{ada}}$ | adaLN modulation layer, $3072\times512$ |
| $\boldsymbol\beta_k,\ \boldsymbol\gamma_k,\ \mathbf g_k$ | shift, scale and gate for sub-block $k$ (MSA = 1, MLP = 2), each length 512 |
| LN | LayerNorm **without** affine parameters |
| MHA | 8-head self-attention with QKV projection $W_{qkv}$ ($1536\times512$) + bias, $d_h = 64$ |
| $W_o,\ \mathbf b_o$ | attention output projection |
| $W_1,\ W_2$ | MLP $512\to2048\to512$ with biases |
| $\operatorname{GELU}_{\tanh}$ | tanh-approximated GELU |
| $\odot$ | elementwise product, broadcast over tokens |
| $Y$ | block output |

**Why "Zero".** DiT initialises $W_{\text{ada}}$ so that $\mathbf g_k = 0$.
Each block then starts as the identity, which stabilises training of very
deep diffusion transformers.

## Approach

| # | Kernel | Fusion |
|---|---|---|
| 1 | modulation GEMM $\operatorname{SiLU}(c)W_{\text{ada}}^{\mathsf T}$ | SiLU applied while loading $c$, bias in the epilogue |
| 2 | LayerNorm + **modulate** | $\operatorname{LN}(x)(1+\gamma_1)+\beta_1$ in one warp-per-row kernel; the sample index selects the modulation row |
| 3 | QKV GEMM (+bias) | – |
| 4 | batched flash attention | grid.z = batch, strided Q/K/V from the packed rows |
| 5 | $W_o$ GEMM | epilogue: $x + g_1\odot(v + b_o)$, the **gated residual** |
| 6 | LayerNorm + modulate | with $\gamma_2, \beta_2$ |
| 7 | FC1 GEMM | epilogue: bias + GELU |
| 8 | FC2 GEMM | epilogue: $x' + g_2\odot(v + b_2)$ |

The epilogue functors receive the output row index. They derive the sample
index $b = \lfloor \text{row}/S\rfloor$ and read the right gate vector, so the
per-sample broadcast costs nothing extra.

## Cost Analysis

$$
W \approx BS\Bigl(2\cdot512\cdot1536 + 2\cdot512^2 + 2\cdot2\cdot512\cdot2048\Bigr) + 4BS^2\cdot512 + 2B\cdot512\cdot3072
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs: QKV, output projection, MLP (per token), attention (quadratic), modulation (per sample) |

With $B = 4$ and $S = 1024$ (a 32 × 32 latent patch grid): about 25 GFLOP of
projections plus 8.6 GFLOP of attention. The GEMMs dominate. Fusion removes
roughly 10 elementwise passes over $B\cdot S\cdot 512$ to $2048$-wide tensors.

## Pitfalls

- **Modulation order.** The 3072-wide output splits as
  `[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]`.
- **$(1 + \gamma)$, not $\gamma$.** The scale is predicted as a residual
  around 1.
- **No affine LayerNorm**: there are no $w$/$b$ vectors to apply.
- **Per-sample broadcast**: row $\to$ sample index mapping in every fused
  kernel.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-3`,
with a raised `cuemu_max_elements` for the large cases.

## Related

- [GPT-2 Block](../074-gpt2-block/), [LLaMA Block](../093-llama-transformer-block/),
  [ViT Patch Embedding](../118-vit-patch-embedding/), [Group Norm](../105-group-normalization/).

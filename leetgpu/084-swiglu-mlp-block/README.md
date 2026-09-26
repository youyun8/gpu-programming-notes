---
title: SwiGLU MLP Block
platform: LeetGPU
upstream: medium/84_swiglu_mlp_block
url: https://leetgpu.com/challenges/swiglu-mlp-block
difficulty: medium
tags: [gemm, fusion, mlp, dual-gemm, llm]
status: solved
---

# SwiGLU MLP Block

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/swiglu-mlp-block)

## Problem

The feed-forward block of LLaMA/Mistral/Gemma. The input
$X \in \mathbb R^{M\times d}$ goes through two parallel projections ("gate"
and "up") to width $d_f$, a SiLU gate, and a down projection back to $d$
($M \le 65\,536$, $d \le 8192$, $d_f \le 32\,768$; benchmark $M = 512$,
$d = 4096$, $d_f = 14\,336$, which is LLaMA-3 8B's MLP; tolerance `1e-4`).
Three GEMMs account for about two-thirds of an LLM's FLOPs. The fusion
opportunity is the elementwise gate between them.

## Formulation

$$
G = XW_g, \qquad U = XW_u, \qquad H = \operatorname{SiLU}(G)\odot U, \qquad Y = HW_d
$$

$$
H_{mj} = \frac{G_{mj}}{1 + e^{-G_{mj}}}\cdot U_{mj}
$$

| Symbol | Meaning |
|---|---|
| $M$ | Number of tokens |
| $d$ | Model width (`d_model`) |
| $d_f$ | Hidden width of the MLP (`d_ffn`, typically $\approx \tfrac83 d$ rounded) |
| $X$ | Input, $M\times d$ |
| $W_g,\ W_u$ | Gate and up projections, $d\times d_f$ (stored (in, out)) |
| $W_d$ | Down projection, $d_f\times d$ |
| $G,\ U$ | Gate and up activations, $M\times d_f$ |
| $\odot$ | Elementwise product |
| $H$ | Gated hidden activations |
| $Y$ | Output, $M\times d$ |

## Approach

### Kernel 1: A Dual GEMM with a Fused Gate

`gemm<true>` computes the tiles of $G$ **and** $U$ at the same time:

- Per $K$-slice, the $64\times16$ tile of $X$ is staged **once**. The
  matching $16\times64$ tiles of $W_g$ and $W_u$ are staged side by side.
- Each thread keeps **two** $4\times4$ accumulator sets (32 registers) and
  does 2 FMAs per $A$ value it loads.
- The epilogue computes $\frac{g}{1+e^{-g}}\cdot u$ and writes only $H$.

Without the dual GEMM: two GEMM launches (reading $X$ twice), writing $G$ and
$U$ ($2Md_f$ floats), and an elementwise kernel that reads them back.

### Kernel 2: $Y = HW_d$

The same register-blocked template in single mode.

## Cost Analysis

$$
W = 2Md\,d_f\cdot 2 + 2Md_f\,d = 6Md\,d_f, \qquad
Q_{\text{saved}} = 4\cdot\bigl(2Md_f\ \text{(write } G, U) + 2Md_f\ \text{(read } G, U) + Md\ \text{(re-read } X)\bigr)
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs: two $d\times d_f$ projections plus one $d_f\times d$ |
| $Q_{\text{saved}}$ | DRAM bytes avoided by the dual-GEMM fusion |

Benchmark: $W = 6\cdot512\cdot4096\cdot14336 \approx 180$ GFLOP, while
$Q_{\text{saved}} \approx 120$ MB. At $M = 512$ the weights dominate the
traffic ($3\cdot 4\cdot d\,d_f = 705$ MB, read about once per 64-row tile
stripe). The kernel is compute-bound on fp32 FMA. A production kernel would
use bf16 tensor cores, and at small $M$ it would split $K$ to fill the GPU.

## Pitfalls

- **Weight layout** (in, out): $XW$, not $XW^{\mathsf T}$.
- **SiLU on the gate only**, then multiply by the *un-activated* up projection.
- **Register pressure.** Two 16-float accumulator sets plus fragments stay
  under the 255-register limit, so there is no spill.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-4`.

## Related

- [SwiGLU](../054-swiglu/), [LLaMA Transformer Block](../093-llama-transformer-block/),
  [GPT-2 Block](../074-gpt2-block/) (GELU MLP), [MoE Top-k Gating](../067-moe-topk-gating/).

---
title: Diffusion Transformer Block
platform: LeetGPU
upstream: hard/116_dit_block
url: https://leetgpu.com/challenges/diffusion-transformer-block
difficulty: hard
tags: [transformer, diffusion, adaln]
status: solved
cuemu_max_elements: 16777216
---

# Diffusion Transformer Block

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/diffusion-transformer-block)

## Problem
A Diffusion-Transformer block with adaLN-Zero conditioning: six modulation
vectors per sample drive two modulated LayerNorms and two gated residuals.

## Approach
`silu(c)·W_adaᵀ` produces the modulation table. The modulation
`LN(x)·(1+scale)+shift` is **fused into the LayerNorm kernel**, and the gated
residual `x + gate·(v + bias)` is fused into the epilogues of the `W_o` and
`W_fc2` GEMMs. Batched attention uses the strided flash kernel with a batch
dimension (`grid.z`).

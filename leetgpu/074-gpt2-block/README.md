---
title: GPT-2 Transformer Block
platform: LeetGPU
upstream: hard/74_gpt2_block
url: https://leetgpu.com/challenges/gpt-2-transformer-block
difficulty: hard
tags: [transformer, gpt-2, gemm, fusion]
status: solved
cuemu_max_elements: 16777216
---

# GPT-2 Transformer Block

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/gpt-2-transformer-block)

## Problem
A full GPT-2 (124M) pre-LN block: LN → QKV → 12-head attention → projection +
residual → LN → FC + GELU(tanh) → projection + residual.

## Approach
Five GEMMs plus attention and LayerNorm, with the elementwise work **fused
into GEMM epilogues** (bias, GELU, residual). Attention reads Q/K/V directly
from the packed `qkv` rows through strides, so no reshape or transpose
kernels are needed. The kernels are a register-blocked GEMM templated on the
epilogue functor and the B layout, a strided flash-attention kernel, and a
warp-per-row LayerNorm.

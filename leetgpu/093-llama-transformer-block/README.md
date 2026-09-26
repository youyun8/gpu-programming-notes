---
title: Llama Transformer Block
platform: LeetGPU
upstream: hard/93_llama_transformer_block
url: https://leetgpu.com/challenges/llama-transformer-block
difficulty: hard
tags: [transformer, llama, gqa, rope, swiglu]
status: solved
---

# Llama Transformer Block

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/llama-transformer-block)

## Problem
A LLaMA block: RMSNorm → Q/K/V → RoPE → causal GQA attention (8/2 heads) →
`W_o` + residual → RMSNorm → SwiGLU FFN + residual.

## Approach
`W_Q, W_K, W_V` are contiguous in the weight buffer, so Q, K and V come from
**one NT-GEMM** with 768 output columns; likewise `W_gate | W_up`. RoPE is
applied in place to the Q and K heads. The attention kernel maps query head
`h` to KV head `h / 4` (GQA) with a causal mask. Residuals are fused into the
`W_o` and `W_down` GEMM epilogues.

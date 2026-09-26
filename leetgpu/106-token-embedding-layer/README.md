---
title: Token Embedding Layer
platform: LeetGPU
upstream: medium/106_token_embedding_layer
url: https://leetgpu.com/challenges/token-embedding-layer
difficulty: medium
tags: [embedding, layernorm, gather]
status: solved
---

# Token Embedding Layer

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/token-embedding-layer)

## Problem
Token + position embedding lookup followed by LayerNorm.

## Approach
One warp per token gathers both rows (coalesced) and keeps the `D/32 ≤ 32`
sums per lane in registers. It then computes the mean and the **centered**
variance from registers (two-pass, no cancellation) and writes the normalized
row. The embedding sum is never written to memory.

---
title: Multi-Head Latent Attention Decode
platform: LeetGPU
upstream: hard/114_multi_head_latent_attention
url: https://leetgpu.com/challenges/multi-head-latent-attention-decode
difficulty: hard
tags: [attention, mla, deepseek, decode]
status: solved
---

# Multi-Head Latent Attention Decode

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/multi-head-latent-attention-decode)

## Problem
DeepSeek-style MLA decode with weight absorption: 128 heads attend over a
compressed latent KV cache (`R ≤ 512`) plus a shared rotary key.

## Approach
The cache row `[c_kv | k_pe]` is exactly the key for the concatenated query
`[q_nope·W_UK | q_pe]`, and its `c_kv` prefix is the value. So after a small
**absorption GEMV**, the core is ordinary attention with a 576-wide score
dimension and a 512-wide value dimension. All heads share the same cache
(MQA-like), so each block's 4 heads reuse every K/V tile. Tiles are streamed
in 128-wide column slices, and a final GEMV per head applies `W_UV`.

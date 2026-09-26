---
title: INT8 KV-Cache Attention
platform: LeetGPU
upstream: medium/96_int8_kv_cache_attention
url: https://leetgpu.com/challenges/int8-kv-cache-attention
difficulty: medium
tags: [attention, decode, flash-decoding, int8]
status: solved
---

# INT8 KV-Cache Attention

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/int8-kv-cache-attention)

## Problem
Decode attention (one query per head) over an int8 KV cache with per-token scales.

## Approach
**Flash-decoding.** One query per head gives too little parallelism, so the
sequence is split into 256-key chunks: grid = (chunks, heads). Each block
computes its chunk's scores (warp per key, dequantized on the fly), a local
softmax `(m, l)`, and `Σ p·V` with coalesced int8 reads. A combine kernel
merges the partials per head with the usual `e^{m_i - m}` rescaling.

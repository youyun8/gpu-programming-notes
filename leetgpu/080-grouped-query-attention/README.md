---
title: Grouped Query Attention
platform: LeetGPU
upstream: medium/80_grouped_query_attention
url: https://leetgpu.com/challenges/grouped-query-attention
difficulty: medium
tags: [attention, gqa, flash-attention]
status: solved
---

# Grouped Query Attention

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/grouped-query-attention)

## Problem
GQA: query head `h` attends with KV head `h / (Hq/Hkv)`; `head_dim ≤ 256`.

## Approach
The warp-per-query flash kernel from [Softmax Attention](../006-softmax-attention),
generalized: a block holds 8 query rows of one head and streams that head's KV
group through shared memory. With `head_dim` up to 256 the tiles need up to
~74 KB, so the shared memory is **dynamic** and opted in with
`cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, ...)`.

Sharing KV heads is also what makes GQA fast in practice: consecutive query
heads reuse the same KV tiles from L2.

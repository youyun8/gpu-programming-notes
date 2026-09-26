---
title: Multi-Head Cross-Attention
platform: LeetGPU
upstream: hard/26_multi_head_cross_attention
url: https://leetgpu.com/challenges/multi-head-cross-attention
difficulty: hard
tags: [attention, cross-attention, flash-attention]
status: solved
---

# Multi-Head Cross-Attention

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/multi-head-cross-attention)

## Problem
Cross-attention with `Q: (M, H, D)` and `K, V: (N, H, D)`, no mask.

## Approach
In `(seq, head, dim)` layout, head `h`'s rows are `H·D` apart and start at
`h·D`, so the reference's transposes are free: the generic strided flash
kernel from [Multi-Head Attention](../012-multi-head-attention) is launched
with those strides and `grid.y = H`.

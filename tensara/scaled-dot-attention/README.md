---
title: Scaled Dot-Product Attention
platform: Tensara
upstream: scaled-dot-attention
url: https://tensara.org/problems/scaled-dot-attention
difficulty: hard
tags: [attention, flash-attention]
status: solved
---

# Scaled Dot-Product Attention

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/scaled-dot-attention)

## Problem
`scaled_dot_product_attention` over `(B, H, S, E)` without a mask.

## Approach
Every `(batch, head)` is an independent attention problem, so `B·H` is folded
into the grid, and the generic flash kernel (lane-per-key scoring, online
softmax, head dimension sliced into 128-wide chunks) runs with head stride `S·E`.

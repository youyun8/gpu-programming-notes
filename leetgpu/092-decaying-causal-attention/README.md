---
title: Decaying Causal Attention
platform: LeetGPU
upstream: medium/92_decaying_causal_attention
url: https://leetgpu.com/challenges/decaying-causal-attention
difficulty: medium
tags: [attention, retention, retnet]
status: solved
---

# Decaying Causal Attention

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/decaying-causal-attention)

## Problem
`out[n] = Σ_{m≤n} γ^{n-m} (Q_n·K_m/√d) V_m` (RetNet's parallel form, no softmax).

## Approach
The flash kernel structure (warp per query row, 32-key shared tiles, lane
per key, shuffled weights) without the softmax bookkeeping. Key tiles stop
at the block's last row, and causality removes half of the work.

---
title: Causal Self-Attention
platform: LeetGPU
upstream: hard/53_casual_attention
url: https://leetgpu.com/challenges/causal-self-attention
difficulty: hard
tags: [attention, causal, flash-attention]
status: solved
---

# Causal Self-Attention

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/causal-self-attention)

## Problem
Causal self-attention (`d ≤ 128`).

## Approach
Flash-style kernel with a per-lane causal mask. Key tiles stop at the block's
last row, which halves the work compared with dense attention.

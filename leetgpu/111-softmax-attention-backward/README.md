---
title: Softmax Attention Backward
platform: LeetGPU
upstream: medium/111_softmax_attention_backward
url: https://leetgpu.com/challenges/softmax-attention-backward
difficulty: medium
tags: [attention, backward, flash-attention]
status: solved
---

# Softmax Attention Backward

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/softmax-attention-backward)

## Problem
Gradients `dQ, dK, dV` of softmax attention.

## Approach
A FlashAttention-2-style backward that never materializes `M×N`:
1. `rowStats` recomputes the forward pass per query row (online softmax) and
   stores `L_i = logsumexp` and `D_i = dO_i·O_i` (= Σ_j P_ij·dP_ij).
2. `gradQ` (query-parallel) computes `dQ_i = Σ_j dS_ij K_j/√d`.
3. `gradKV` (key-parallel) computes `dV_j = Σ_i P_ij dO_i` and
   `dK_j = Σ_i dS_ij Q_i/√d`,

where `P = exp(S − L)` and `dS = P ⊙ (dP − D)`. Splitting dQ from dK/dV
avoids atomics: each output row is owned by exactly one warp.

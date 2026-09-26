---
title: Categorical Cross Entropy Loss
platform: LeetGPU
upstream: medium/25_categorical_cross_entropy_loss
url: https://leetgpu.com/challenges/categorical-cross-entropy-loss
difficulty: medium
tags: [reduction, logsumexp, warp-per-row]
status: solved
---

# Categorical Cross Entropy Loss

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/categorical-cross-entropy-loss)

## Problem
Mean over `N` samples of `logsumexp(z_j) - z_j[y_j]`.

## Approach
One warp per sample. Each lane keeps an online `(max, Σexp)` pair over its
strided logits, and the pairs are merged with `__shfl_xor_sync`. Lane 0 adds
the sample loss to an fp64 per-block sum; a final kernel divides by `N`.

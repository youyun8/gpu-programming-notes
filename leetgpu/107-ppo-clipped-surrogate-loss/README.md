---
title: PPO Clipped Surrogate Loss
platform: LeetGPU
upstream: medium/107_ppo_clipped_surrogate_loss
url: https://leetgpu.com/challenges/ppo-clipped-surrogate-loss
difficulty: medium
tags: [reduction, rl]
status: solved
---

# PPO Clipped Surrogate Loss

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/ppo-clipped-surrogate-loss)

## Problem
`−mean(min(r·A, clip(r, 1±ε)·A))`, `r = exp(log π − log π_old)`.

## Approach
Elementwise transform fused into a two-pass reduction.

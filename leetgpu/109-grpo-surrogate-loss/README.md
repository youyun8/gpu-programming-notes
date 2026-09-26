---
title: GRPO Surrogate Loss
platform: LeetGPU
upstream: medium/109_grpo_surrogate_loss
url: https://leetgpu.com/challenges/grpo-surrogate-loss
difficulty: medium
tags: [reduction, rl]
status: solved
---

# GRPO Surrogate Loss

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/grpo-surrogate-loss)

## Problem
GRPO: group-normalized advantages + clipped surrogate − β·KL(k3), averaged.

## Approach
A small kernel computes per-group advantages (population std, as in the
reference); a grid-stride reduction then evaluates the token terms, reading
`A[b,g]` via `i / S`.

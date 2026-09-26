---
title: Parallel Reverse Scan (GAE)
platform: LeetGPU
upstream: medium/110_gae_reverse_scan
url: https://leetgpu.com/challenges/parallel-reverse-scan-gae
difficulty: medium
tags: [scan, rl]
status: solved
---

# Parallel Reverse Scan (GAE)

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/parallel-reverse-scan-gae)

## Problem
Generalized Advantage Estimation: `A_t = δ_t + γλ·A_{t+1}` (right-to-left).

## Approach
The same affine-map scan as [Linear Recurrence](../082-linear-recurrence), run from
the end of the sequence: thread `k` owns the k-th chunk counted from the right,
so the block scan follows the direction of the recurrence.

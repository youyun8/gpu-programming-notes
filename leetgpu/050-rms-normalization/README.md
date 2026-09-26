---
title: RMS Normalization
platform: LeetGPU
upstream: medium/50_rms_normalization
url: https://leetgpu.com/challenges/rms-normalization
difficulty: medium
tags: [normalization, reduction]
status: solved
---

# RMS Normalization

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/rms-normalization)

## Problem
`y = γ·x / sqrt(mean(x²) + ε) + β` over one vector.

## Approach
Three passes: block partial sums of `x²` (fp64), one block computes `1/rms`
into a `__device__` variable, and an elementwise scale and shift.

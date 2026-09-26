---
title: Group Normalization
platform: LeetGPU
upstream: medium/105_group_normalization
url: https://leetgpu.com/challenges/group-normalization
difficulty: medium
tags: [normalization]
status: solved
---

# Group Normalization

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/group-normalization)

## Problem
GroupNorm over `(N, C, H, W)`.

## Approach
In NCHW layout, the elements of one `(n, group)` are contiguous, so each
group is a single block reduction (sum and sum of squares in fp64), followed by
normalization with per-channel γ/β.

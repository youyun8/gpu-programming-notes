---
title: Value Clipping
platform: LeetGPU
upstream: easy/62_value_clipping
url: https://leetgpu.com/challenges/value-clipping
difficulty: easy
tags: [elementwise]
status: solved
---

# Value Clipping

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/value-clipping)

## Problem
Clamp every element to `[lo, hi]`.

## Approach
`fminf(fmaxf(x, lo), hi)` — branch-free elementwise kernel.

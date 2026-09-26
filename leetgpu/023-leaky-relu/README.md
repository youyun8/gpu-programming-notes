---
title: Leaky ReLU
platform: LeetGPU
upstream: easy/23_leaky_relu
url: https://leetgpu.com/challenges/leaky-relu
difficulty: easy
tags: [elementwise, activation, vectorized]
status: solved
---

# Leaky ReLU

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/leaky-relu)

## Problem
`f(x) = x` for `x > 0`, else `0.01·x`.

## Approach
Same `float4` elementwise template as ReLU; a select, no branch divergence.

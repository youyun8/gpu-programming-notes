---
title: ReLU
platform: LeetGPU
upstream: easy/21_relu
url: https://leetgpu.com/challenges/relu
difficulty: easy
tags: [elementwise, activation, vectorized]
status: solved
---

# ReLU

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/relu)

## Problem
`output = max(0, input)` elementwise.

## Approach
`float4` vectorized elementwise kernel with a scalar tail; `fmaxf` compiles to
a single instruction.

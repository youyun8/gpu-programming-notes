---
title: Weight Dequantization
platform: LeetGPU
upstream: medium/64_weight_dequantization
url: https://leetgpu.com/challenges/weight-dequantization
difficulty: medium
tags: [elementwise, quantization]
status: solved
---

# Weight Dequantization

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/weight-dequantization)

## Problem
`Y = X ⊙ S` with one scale per `T×T` tile.

## Approach
A 2-D elementwise kernel with `threadIdx.x` along columns. Each scale value is
shared by `T²` threads and always hits cache.

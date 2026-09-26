---
title: FP16 Dot Product
platform: LeetGPU
upstream: medium/58_fp16_dot_product
url: https://leetgpu.com/challenges/fp16-dot-product
difficulty: medium
tags: [reduction, fp16]
status: solved
---

# FP16 Dot Product

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/fp16-dot-product)

## Problem
Dot product of two fp16 vectors with fp32 accumulation, returned as fp16.

## Approach
`half2` loads (two elements per 32-bit transaction) are converted with
`__half22float2` and accumulated with `fmaf`; then fp64 block partials and a
final block. An odd trailing element is handled by one thread.

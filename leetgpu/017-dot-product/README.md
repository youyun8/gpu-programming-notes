---
title: Dot Product
platform: LeetGPU
upstream: medium/17_dot_product
url: https://leetgpu.com/challenges/dot-product
difficulty: medium
tags: [reduction, two-pass]
status: solved
---

# Dot Product

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/dot-product)

## Problem
`result[0] = Σ A[i]·B[i]`.

## Approach
Same two-pass scheme as [Reduction](../004-reduction): `float4` loads and
`fmaf` per thread, fp64 block partials, one final block.

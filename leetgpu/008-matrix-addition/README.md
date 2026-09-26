---
title: Matrix Addition
platform: LeetGPU
upstream: easy/8_matrix_addition
url: https://leetgpu.com/challenges/matrix-addition
difficulty: easy
tags: [elementwise, vectorized]
status: solved
---

# Matrix Addition

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/matrix-addition)

## Problem
`C = A + B` for `N×N` float32 matrices.

## Approach
The matrices are contiguous, so this is a flat vector add over `N²` elements.
Use `float4` loads/stores for the bulk and let the first `N² mod 4` threads
handle the scalar tail.

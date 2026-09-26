---
title: Matrix Copy
platform: LeetGPU
upstream: easy/31_matrix_copy
url: https://leetgpu.com/challenges/matrix-copy
difficulty: easy
tags: [memory-bound, vectorized]
status: solved
---

# Matrix Copy

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/matrix-copy)

## Problem
Copy an `N×N` float32 matrix.

## Approach
A bandwidth benchmark: flat `float4` copy (16-byte transactions) plus a scalar
tail. (`cudaMemcpy` device-to-device would also work, but writing the kernel
is the point.)

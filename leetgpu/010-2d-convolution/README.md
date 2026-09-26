---
title: 2D Convolution
platform: LeetGPU
upstream: medium/10_2d_convolution
url: https://leetgpu.com/challenges/2d-convolution
difficulty: medium
tags: [convolution, shared-memory, tiling]
status: solved
---

# 2D Convolution

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/2d-convolution)

## Problem
"Valid" 2D cross-correlation, input up to 3072², kernel up to 31×31.

## Approach
A 32×8 block computes a 32×32 output tile, 4 rows per thread. The input window
(`(32+kr-1) × (32+kc-1)`, up to 62×62) and the kernel are staged in dynamic
shared memory. In the inner loop, consecutive threads read consecutive shared
words, and every kernel weight is a broadcast.

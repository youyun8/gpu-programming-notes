---
title: 2D FFT
platform: LeetGPU
upstream: medium/78_2d_fft
url: https://leetgpu.com/challenges/2d-fft
difficulty: medium
tags: [fft, transpose, shared-memory]
status: solved
---

# 2D FFT

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/2d-fft)

## Problem
Complex 2-D DFT of an `M×N` signal (interleaved re/im).

## Approach
Row-column decomposition with transposes, so every pass works on contiguous rows:
row FFTs → transpose → row FFTs → transpose back. A row transform runs in
one block in shared memory: radix-2 Cooley–Tukey (bit-reversed load via
`__brev`, then log₂n butterfly stages) for power-of-two lengths, and a direct
DFT otherwise (the tests only use small non-power-of-two sizes). Twiddles use
`sincospif(2k/n)` with `k mod n` reduced in integer arithmetic, so the
angles stay exact for large `n`.

## Pitfalls
- A direct DFT is `O(n²)`; supporting large non-power-of-two sizes would need
  Bluestein's algorithm.

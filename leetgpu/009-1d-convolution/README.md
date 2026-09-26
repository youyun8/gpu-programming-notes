---
title: 1D Convolution
platform: LeetGPU
upstream: easy/9_1d_convolution
url: https://leetgpu.com/challenges/1d-convolution
difficulty: easy
tags: [convolution, shared-memory, dynamic-shared-memory]
status: solved
---

# 1D Convolution

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/1d-convolution)

## Problem
"Valid" 1D cross-correlation: `output[i] = Σ_j input[i+j] · kernel[j]`,
output length `input_size - kernel_size + 1`, kernel up to 2047 taps.

## Approach
Each block produces 1024 outputs (4 per thread). It stages the whole kernel and
its input window (`1024 + kernel_size - 1` values) in **dynamic** shared
memory, so every input element is read from DRAM ~once instead of
`kernel_size` times. Thread `t` computes outputs `t, t+256, t+512, t+768`:
in the inner loop neighbouring threads read neighbouring shared words (no
bank conflicts) and `kernel[j]` is a broadcast.

## Pitfalls
- Dynamic shared memory size depends on `kernel_size` — pass it as the third
  launch parameter (~17 KB at the maximum size, under the 48 KB default).
- Out-of-range window elements are zero-filled; they only feed outputs that
  are never written.

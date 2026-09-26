---
title: Fast Fourier Transform
platform: LeetGPU
upstream: hard/39_Fast_Fourier_transform
url: https://leetgpu.com/challenges/fast-fourier-transform
difficulty: hard
tags: [fft, bluestein, stockham]
status: solved
---

# Fast Fourier Transform

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/fast-fourier-transform)

## Problem
1-D complex FFT of any length `N ≤ 262,144`.

## Approach
- **Power-of-two `N`:** Stockham radix-2, with `log₂N` passes in global memory.
  Stockham is self-sorting, so no bit-reversal permutation is needed.
- **Any other `N`:** **Bluestein's chirp-z algorithm**. With
  `w_n = e^{-iπn²/N}`, the DFT becomes a convolution,
  `X_k = w_k · Σ_n (x_n w_n) · conj(w_{k−n})`, which is evaluated with
  power-of-two FFTs of size `L ≥ 2N−1`.

Chirp phases reduce `n² mod 2N` in 64-bit integers before `sincospif`, so they
stay exact at large `n`. Tested against numpy for `N = 1…262,144`, including
primes and `2¹⁸−1`.

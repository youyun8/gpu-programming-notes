---
title: NVFP4 GEMV
platform: Tensara
upstream: nvfp4-gemv
url: https://tensara.org/problems/nvfp4-gemv
difficulty: hard
tags: [gemv, quantization, nvfp4]
status: solved
---

# NVFP4 GEMV

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/nvfp4-gemv)

## Problem
fp16 `y = A·x` with NVFP4 `A` and `x`.

## Approach
A bandwidth-bound GEMV at ~0.56 bytes per weight: one warp per row, and each
lane decodes whole 16-element blocks (8 bytes + 1 scale), accumulating the
block's integer-like products before applying both block scales.

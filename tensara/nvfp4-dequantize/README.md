---
title: NVFP4 Dequantization
platform: Tensara
upstream: nvfp4-dequantize
url: https://tensara.org/problems/nvfp4-dequantize
difficulty: medium
tags: [quantization, nvfp4, low-precision]
status: solved
---

# NVFP4 Dequantization

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/nvfp4-dequantize)

## Problem
Decode NVFP4: E2M1 values, one E4M3 scale per 16 (swizzled layout), and an
fp32 global scale.

## Approach
Each thread decodes a byte into two values: `e2m1 · e4m3(scale) / sf_g`.

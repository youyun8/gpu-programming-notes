---
title: NVFP4 Quantization
platform: Tensara
upstream: nvfp4-quantize
url: https://tensara.org/problems/nvfp4-quantize
difficulty: medium
tags: [quantization, nvfp4, low-precision]
status: solved
---

# NVFP4 Quantization

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/nvfp4-quantize)

## Problem
fp16 → NVFP4 with flashinfer semantics: `sf = e4m3(sf_g·amax/6)`, `q = e2m1(x·sf_g/sf)`.

## Approach
Half a warp per 16-element block (16-lane shuffle max). Scales are written in
the swizzled 128×4 layout, with padding zeroed by a first kernel.

## Pitfalls
- Tested locally against a Python re-implementation of flashinfer's semantics
  (`tools/cuemu/lowp_reference.py`), because flashinfer itself is CUDA-only.

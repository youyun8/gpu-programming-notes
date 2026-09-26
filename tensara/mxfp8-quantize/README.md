---
title: MXFP8 Quantization
platform: Tensara
upstream: mxfp8-quantize
url: https://tensara.org/problems/mxfp8-quantize
difficulty: medium
tags: [quantization, mxfp8, low-precision]
status: solved
---

# MXFP8 Quantization

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/mxfp8-quantize)

## Problem
fp32 → MXFP8 (E4M3), matching TorchAO `to_mx`.

## Approach
As for MXFP4, but the target exponent is `floor(log2 amax) − 8`, and elements
are clamped to ±448 and rounded to E4M3 (nearest even).

---
title: MXFP4 Dequantization
platform: Tensara
upstream: mxfp4-dequantize
url: https://tensara.org/problems/mxfp4-dequantize
difficulty: easy
tags: [quantization, mxfp4, low-precision]
status: solved
---

# MXFP4 Dequantization

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/mxfp4-dequantize)

## Problem
Decode OCP MXFP4 (E2M1 values, one E8M0 scale per 32) to fp32.

## Approach
Each thread expands one byte (two FP4 values, element 2i in the low nibble)
and writes a `float2`. Formats are decoded with integer bit manipulation
(see `lowp` helpers in the source), so no `cuda_fp4.h` or architecture
dependency is needed.

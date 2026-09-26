---
title: MXFP8 Dequantization
platform: Tensara
upstream: mxfp8-dequantize
url: https://tensara.org/problems/mxfp8-dequantize
difficulty: easy
tags: [quantization, mxfp8, low-precision]
status: solved
---

# MXFP8 Dequantization

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/mxfp8-dequantize)

## Problem
Decode MXFP8 (E4M3 values, E8M0 scale per 32) to fp32.

## Approach
An elementwise decode: E4M3 has bias 7, subnormals below 2⁻⁶, and
`0x7F`/`0xFF` are NaN. The scale is `2^(b−127)`.

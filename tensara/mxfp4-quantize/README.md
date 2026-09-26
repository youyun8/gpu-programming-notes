---
title: MXFP4 Quantization
platform: Tensara
upstream: mxfp4-quantize
url: https://tensara.org/problems/mxfp4-quantize
difficulty: medium
tags: [quantization, mxfp4, low-precision]
status: solved
---

# MXFP4 Quantization

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/mxfp4-quantize)

## Problem
fp32 → MXFP4, matching TorchAO `to_mx` (checked after dequantization).

## Approach
One warp per 32-element block. The amax is a warp max, and the scale exponent
is `floor(log2 amax) − 2` read straight from the float bits (TorchAO's FLOOR
mode). Each lane encodes `x / 2^e` to E2M1 (round to nearest even,
saturating), and pairs of lanes are packed with a shuffle.

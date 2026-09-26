---
title: MXFP4 GEMM
platform: Tensara
upstream: mxfp4-gemm
url: https://tensara.org/problems/mxfp4-gemm
difficulty: hard
tags: [gemm, quantization, mxfp4, block-scaling]
status: solved
---

# MXFP4 GEMM

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/mxfp4-gemm)

## Problem
`C = A·Bᵀ` with MXFP4 operands and **swizzled** E8M0 scales (the 128×4
"32_4_4" layout used by block-scaled tensor-core MMA).

## Approach
A register-blocked SGEMM that **dequantizes while staging tiles into shared
memory**, reading each scale through the swizzle index
`((r/128)·⌈cols/4⌉ + c/4)·512 + (r%32)·16 + (r/32%4)·4 + c%4`. The quantized data
is read once per tile, a 4–8× smaller DRAM footprint than fp32.

## Pitfalls
- The swizzle formula was verified against TorchAO's `to_blocked`.

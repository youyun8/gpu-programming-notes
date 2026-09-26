---
title: MXFP8 GEMM
platform: Tensara
upstream: mxfp8-gemm
url: https://tensara.org/problems/mxfp8-gemm
difficulty: hard
tags: [gemm, quantization, mxfp8, block-scaling]
status: solved
---

# MXFP8 GEMM

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/mxfp8-gemm)

## Problem
`C = A·Bᵀ` with MXFP8 operands and swizzled E8M0 scales.

## Approach
Same structure as [mxfp4-gemm](../mxfp4-gemm) with E4M3 decoding.

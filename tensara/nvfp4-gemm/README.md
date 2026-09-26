---
title: NVFP4 GEMM
platform: Tensara
upstream: nvfp4-gemm
url: https://tensara.org/problems/nvfp4-gemm
difficulty: hard
tags: [gemm, quantization, nvfp4, block-scaling]
status: solved
---

# NVFP4 GEMM

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/nvfp4-gemm)

## Problem
fp16 `C = A·Bᵀ` with NVFP4 operands (16-element E4M3 block scales + global scales).

## Approach
The dequantize-on-load SGEMM from [mxfp4-gemm](../mxfp4-gemm) with 16-element
blocks. Both global scales are applied once in the epilogue before the fp16
conversion.

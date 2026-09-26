---
title: INT4 Weight-Only Quantized MatMul
platform: LeetGPU
upstream: medium/81_int4_matmul
url: https://leetgpu.com/challenges/int4-weight-only-quantized-matmul
difficulty: medium
tags: [gemm, int4, quantization, tensor-cores, wmma]
status: solved
---

# INT4 Weight-Only Quantized MatMul

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/int4-weight-only-quantized-matmul)

## Problem
W4A16: `y = x · Wᵀ`, where `W` is packed int4 (two per byte, offset 8) with
one fp16 scale per `group_size` weights.

## Approach
`W` stays packed in global memory (0.5 byte per weight, which is the whole
point of weight-only quantization) and is **dequantized on the fly** while its
tile is staged into shared memory as fp16. The tensor cores (WMMA fp16, fp32
accumulate) then see an ordinary tile. Since `W` is `N×K` row-major, `Wᵀ` is
column-major, so the B fragment is loaded with `wmma::col_major` directly from
the `[n][k]` tile, without a transpose.

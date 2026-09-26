---
title: INT8 Quantized MatMul
platform: LeetGPU
upstream: medium/32_int8_quantized_matmul
url: https://leetgpu.com/challenges/int8-quantized-matmul
difficulty: medium
tags: [gemm, int8, quantization, tensor-cores, wmma]
status: solved
---

# INT8 Quantized MatMul

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/int8-quantized-matmul)

## Problem
Quantized GEMM: `clamp(round(Σ(A-zA)(B-zB) · sA·sB/sC) + zC, -128, 127)`.
Exact match required.

## Approach
`(A - zA)` doesn't fit in int8, so the zero points are **folded out**:

    Σ(A-zA)(B-zB) = A·B − zB·rowsum(A) − zA·colsum(B) + K·zA·zB

- The raw int8 `A·B` runs on tensor cores (WMMA `s8 × s8 → s32`).
- Two small pre-pass kernels compute `rowsum(A)` and `colsum(B)`.
- The epilogue applies the correction and requantizes **in the same float32
  operation order as the reference** (`(acc·sA)·sB / sC`, then
  round-half-to-even with `rintf`), because the check is bit-exact.

## Pitfalls
- WMMA needs 32-byte-aligned fragment pointers. With 1-byte elements, a
  16-column offset inside a padded row is only 16-byte aligned, so the shared
  tiles are stored as contiguous 16×16 blocks.
- `torch.round` rounds half to even; `roundf` rounds half away from zero.

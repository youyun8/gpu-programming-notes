---
title: FP16 Batched Matrix Multiplication
platform: LeetGPU
upstream: medium/57_fp16_batched_matmul
url: https://leetgpu.com/challenges/fp16-batched-matrix-multiplication
difficulty: medium
tags: [gemm, fp16, tensor-cores, wmma, batched]
status: solved
---

# FP16 Batched Matrix Multiplication

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/fp16-batched-matrix-multiplication)

## Problem
Batched fp16 GEMM with fp32 accumulation.

## Approach
The WMMA kernel from [FP16 GEMM](../022-gemm) (64×64 block tile, 2×2 16×16×16
fragments per warp, zero-padded shared tiles), with the batch on `gridDim.z`.

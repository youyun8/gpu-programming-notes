---
title: Batched Matrix Multiplication
platform: LeetGPU
upstream: medium/30_batched_matrix_multiplication
url: https://leetgpu.com/challenges/batched-matrix-multiplication
difficulty: medium
tags: [gemm, batched, register-blocking]
status: solved
---

# Batched Matrix Multiplication

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/batched-matrix-multiplication)

## Problem
`C[b] = A[b] · B[b]` for a batch of fp32 matrices.

## Approach
The 64×64 register-blocked SGEMM from [Matrix Multiplication](../002-matrix-multiplication),
with the batch index on `gridDim.z` and per-batch pointer offsets.

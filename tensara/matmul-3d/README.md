---
title: 3D Tensor-Matrix Multiplication
platform: Tensara
upstream: matmul-3d
url: https://tensara.org/problems/matmul-3d
difficulty: hard
tags: [gemm, batched]
status: solved
---

# 3D Tensor-Matrix Multiplication

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/matmul-3d)

## Problem
`(N, M, K) × (K, L)`.

## Approach
The leading dims of A are contiguous, so this is one `(N·M) × K × L` SGEMM.

---
title: 4D Tensor-Matrix Multiplication
platform: Tensara
upstream: matmul-4d
url: https://tensara.org/problems/matmul-4d
difficulty: hard
tags: [gemm, batched]
status: solved
---

# 4D Tensor-Matrix Multiplication

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/matmul-4d)

## Problem
`einsum("bijl,lk->bijk")`.

## Approach
One `(b·i·j) × l × k` SGEMM.

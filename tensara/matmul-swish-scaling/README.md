---
title: Matrix Multiplication with Swish and Scaling
platform: Tensara
upstream: matmul-swish-scaling
url: https://tensara.org/problems/matmul-swish-scaling
difficulty: medium
tags: [gemm, fusion]
status: solved
---

# Matrix Multiplication with Swish and Scaling

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/matmul-swish-scaling)

## Problem
`scale · swish(A·B)`.

## Approach
SGEMM with a fused epilogue.

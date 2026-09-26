---
title: Matrix Multiplication with Swish Activation
platform: Tensara
upstream: matmul-swish
url: https://tensara.org/problems/matmul-swish
difficulty: medium
tags: [gemm, fusion]
status: solved
---

# Matrix Multiplication with Swish Activation

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/matmul-swish)

## Problem
`s · swish(x·Wᵀ + bias)`.

## Approach
An NT-GEMM (W is stored `(out, in)`) with bias, swish and scaling fused into
the epilogue.

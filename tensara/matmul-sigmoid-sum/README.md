---
title: Matrix Multiplication with Sigmoid and Sum
platform: Tensara
upstream: matmul-sigmoid-sum
url: https://tensara.org/problems/matmul-sigmoid-sum
difficulty: medium
tags: [gemm, fusion, reduction]
status: solved
---

# Matrix Multiplication with Sigmoid and Sum

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/matmul-sigmoid-sum)

## Problem
`Σ sigmoid(A·B)` as a scalar.

## Approach
The product is never stored: the SGEMM epilogue applies the sigmoid, each
block reduces its tile, and one fp64 `atomicAdd` per block accumulates the
total.

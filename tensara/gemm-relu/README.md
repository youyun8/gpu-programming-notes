---
title: GEMM with Bias and ReLU
platform: Tensara
upstream: gemm-relu
url: https://tensara.org/problems/gemm-relu
difficulty: medium
tags: [gemm, fusion]
status: solved
---

# GEMM with Bias and ReLU

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/gemm-relu)

## Problem
`relu(A·Wᵀ + b)`.

## Approach
NT-GEMM with a fused bias + ReLU epilogue.

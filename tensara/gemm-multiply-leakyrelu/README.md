---
title: GEMM with Element-wise Multiply and LeakyReLU
platform: Tensara
upstream: gemm-multiply-leakyrelu
url: https://tensara.org/problems/gemm-multiply-leakyrelu
difficulty: medium
tags: [gemm, fusion]
status: solved
---

# GEMM with Element-wise Multiply and LeakyReLU

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/gemm-multiply-leakyrelu)

## Problem
`leaky_relu((A·B) ⊙ C, α)`.

## Approach
SGEMM whose epilogue reads `C`, multiplies, and applies the activation.

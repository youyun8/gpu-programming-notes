---
title: Symmetric Matrix Multiplication
platform: Tensara
upstream: symmetric-matmul
url: https://tensara.org/problems/symmetric-matmul
difficulty: medium
tags: [gemm]
status: solved
---

# Symmetric Matrix Multiplication

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/symmetric-matmul)

## Problem
Product of two symmetric matrices.

## Approach
Symmetry doesn't reduce the FLOPs of a dense product (it only means `Bᵀ = B`),
so this is the plain SGEMM.

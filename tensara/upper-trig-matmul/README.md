---
title: Upper Triangular Matrix Multiplication
platform: Tensara
upstream: upper-trig-matmul
url: https://tensara.org/problems/upper-trig-matmul
difficulty: medium
tags: [gemm, triangular]
status: solved
---

# Upper Triangular Matrix Multiplication

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/upper-trig-matmul)

## Problem
Product of two upper-triangular matrices.

## Approach
The mirror image of [lower-trig-matmul](../lower-trig-matmul): tiles below the
diagonal are zero, and the K range is `[row0, col0 + 64)`.

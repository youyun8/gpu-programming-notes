---
title: Matrix Vector Multiplication
platform: Tensara
upstream: matrix-vector
url: https://tensara.org/problems/matrix-vector
difficulty: easy
tags: [gemv]
status: solved
---

# Matrix Vector Multiplication

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/matrix-vector)

## Problem
`c = A·b`.

## Approach
GEMV is bandwidth-bound: one warp per row, `float4` loads when rows are
16-byte aligned, and a shuffle reduction.

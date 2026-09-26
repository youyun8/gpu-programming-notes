---
title: Lower Triangular Matrix Multiplication
platform: Tensara
upstream: lower-trig-matmul
url: https://tensara.org/problems/lower-trig-matmul
difficulty: medium
tags: [gemm, triangular]
status: solved
---

# Lower Triangular Matrix Multiplication

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/lower-trig-matmul)

## Problem
Product of two lower-triangular matrices.

## Approach
`C[i][j]` only needs `k ∈ [j, i]`. The SGEMM skips output tiles above the
diagonal (writing zeros) and limits each tile's K loop to
`[col0, row0 + 64)`, roughly a third of the dense FLOPs. Loads mask the upper
triangle, just as the reference applies `tril`.

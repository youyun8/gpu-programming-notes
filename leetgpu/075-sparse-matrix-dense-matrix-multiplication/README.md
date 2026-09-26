---
title: Sparse Matrix-Dense Matrix Multiplication
platform: LeetGPU
upstream: medium/75_sparse_matrix_dense_matrix_multiplication
url: https://leetgpu.com/challenges/sparse-matrix-dense-matrix-multiplication
difficulty: medium
tags: [gemm, sparsity]
status: solved
---

# Sparse Matrix-Dense Matrix Multiplication

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/sparse-matrix-dense-matrix-multiplication)

## Problem
`C = A·B` where `A` is 60–70% zeros but stored densely.

## Approach
At ~35% density a dense register-blocked SGEMM beats sparse formats on a
GPU: CSR would need a conversion pass, and its irregular gathers from `B` cost
more than the ~2× FLOPs saved. Sparse kernels pay off below roughly 1–5% density.

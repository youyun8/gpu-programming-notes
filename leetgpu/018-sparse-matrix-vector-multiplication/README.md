---
title: Sparse Matrix-Vector Multiplication
platform: LeetGPU
upstream: medium/18_sparse_matrix_vector_multiplication
url: https://leetgpu.com/challenges/sparse-matrix-vector-multiplication
difficulty: medium
tags: [gemv, warp-per-row]
status: solved
---

# Sparse Matrix-Vector Multiplication

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/sparse-matrix-vector-multiplication)

## Problem
`y = A x` where `A` (`M×N`, 60-70% zeros) is stored **densely**.

## Approach
Because the matrix is dense in memory, this is a bandwidth-bound GEMV:
there is no index structure to exploit, and skipping zeros saves FLOPs but no
bytes. One warp per row: lanes stride the columns (coalesced 128-byte
reads), `x` (≤40 KB) stays in cache, and a shuffle reduction produces `y[row]`.

## Pitfalls
- Converting to CSR on the GPU first costs a full pass over `A` plus a scan.
  It only pays off when the matrix is reused.

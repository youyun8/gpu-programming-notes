---
title: Matrix Multiplication
platform: Tensara
upstream: matrix-multiplication
url: https://tensara.org/problems/matrix-multiplication
difficulty: medium
tags: [gemm, register-blocking]
status: solved
---

# Matrix Multiplication

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/matrix-multiplication)

## Problem
`C = A·B` (fp32, row-major).

## Approach
Register-blocked SGEMM: a 64×64 block tile, 16-wide K slices in shared memory
(A stored transposed), and 4×4 outputs per thread with stride 16 so the
stores are coalesced. See [tutorial 04](../../tutorials/04-tiled-matmul.md).
The same kernel, templated on B's layout and a fused epilogue functor, is
reused by the other GEMM problems below.

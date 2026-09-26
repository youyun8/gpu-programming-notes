---
title: Matrix Multiplication
platform: LeetGPU
upstream: easy/2_matrix_multiplication
url: https://leetgpu.com/challenges/matrix-multiplication
difficulty: easy
tags: [gemm, shared-memory, tiling, register-blocking]
status: solved
---

# Matrix Multiplication

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/matrix-multiplication)

## Problem
`C (M×K) = A (M×N) · B (N×K)`, all float32 row-major. Note that the inner
dimension is called `N` here.

## Approach
Classic shared-memory + register tiling (see [tutorial 04](../../tutorials/04-tiled-matmul.md)):
- A 256-thread block computes a 64×64 tile of `C`; each thread owns a 4×4
  sub-tile laid out with stride 16 so that neighbouring threads write
  neighbouring columns (coalesced stores).
- The inner dimension is walked in 16-wide slices. The `A` slice is stored
  **transposed** in shared memory so both operands are read along rows; the
  `+4` padding avoids bank conflicts on the transposed store.
- Each shared-memory value is reused 4 times from registers, i.e. 16 FMAs per
  8 shared loads instead of 1 FMA per 2 loads in the naive tiled kernel.

## Pitfalls
- Edge tiles load zeros instead of skipping the load, so the barriers stay uniform.
- Two `__syncthreads()` per K-slice: after loading and before overwriting.

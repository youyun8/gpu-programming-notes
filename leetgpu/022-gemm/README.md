---
title: General Matrix Multiplication (GEMM)
platform: LeetGPU
upstream: medium/22_gemm
url: https://leetgpu.com/challenges/general-matrix-multiplication-gemm
difficulty: medium
tags: [gemm, fp16, tensor-cores, wmma]
status: solved
---

# General Matrix Multiplication (GEMM)

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/general-matrix-multiplication-gemm)

## Problem
`C = α·A·B + β·C` in fp16 with fp32 accumulation; `16 ≤ M, N, K ≤ 4096`
(not necessarily multiples of 16).

## Approach
Tensor cores via the `nvcuda::wmma` API:
- A 128-thread block computes a 64×64 tile of `C`. Each warp owns a 32×32
  quadrant = 2×2 fragments of 16×16×16.
- 64×32 and 32×64 slices of `A` and `B` are staged in shared memory, with
  zero padding at the matrix edges, so arbitrary sizes work.
- Epilogue: the accumulators go to a shared fp32 tile, then every thread
  applies `α·acc + β·C_old` and converts to fp16, with bounds checks.

## Pitfalls
- WMMA pointers must be 32-byte aligned, and `ldm` must be a multiple of 16
  bytes. The shared-memory pitches (40 and 72 halves, 68 floats) are chosen to
  satisfy this and to stagger the banks.
- Read `C_old` **before** overwriting it: β uses the initial `C`.

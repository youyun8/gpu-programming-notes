---
title: Ordinary Least Squares
platform: LeetGPU
upstream: medium/33_ordinary_least_squares
url: https://leetgpu.com/challenges/ordinary-least-squares
difficulty: medium
tags: [linear-algebra, cholesky, normal-equations, fp64]
status: solved
---

# Ordinary Least Squares

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/ordinary-least-squares)

## Problem
Least-squares coefficients `β = (XᵀX)⁻¹ Xᵀy` for `X: n×F` (F ≤ 1000).

## Approach
1. `XᵀX` is an "Aᵀ·A" GEMM: 16×16 tiles of the sample dimension are staged in
   shared memory, with fp64 accumulation. `Xᵀy` uses one thread per feature,
   coalesced across features.
2. A single 1024-thread block computes a right-looking Cholesky factorization
   `XᵀX = LLᵀ` in place (pivot, column scale, trailing rank-1 update, one
   barrier per step), then does forward/back substitution with a block
   reduction per row.

fp64 matters because forming `XᵀX` squares the condition number of `X`.

## Pitfalls
- A QR-based solve is more robust in general; Cholesky on the normal equations
  is what the reference does and is fine for full-rank inputs.

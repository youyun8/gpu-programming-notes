---
title: SwiGLU MLP Block
platform: LeetGPU
upstream: medium/84_swiglu_mlp_block
url: https://leetgpu.com/challenges/swiglu-mlp-block
difficulty: medium
tags: [gemm, fusion, mlp]
status: solved
---

# SwiGLU MLP Block

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/swiglu-mlp-block)

## Problem
`(SiLU(x·Wg) ⊙ (x·Wu)) · Wd`.

## Approach
1. A **dual GEMM** computes the gate and up tiles in the same kernel, loading
   the `x` tile once and keeping two accumulator sets in registers. The
   epilogue applies `silu(g)·u`, so gate and up are never written to memory.
2. `H · Wd` uses the same register-blocked SGEMM.

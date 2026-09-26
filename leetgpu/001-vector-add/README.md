---
title: Vector Addition
platform: LeetGPU
upstream: easy/1_vector_add
url: https://leetgpu.com/challenges/vector-addition
difficulty: easy
tags: [elementwise, memory-bound]
status: solved
---

# Vector Addition

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/vector-addition)

## Problem
`C[i] = A[i] + B[i]` for float32 vectors of length `N` (device pointers).

## Approach
One thread per element: `i = blockIdx.x * blockDim.x + threadIdx.x`, 256-thread
blocks, `ceil(N / 256)` blocks and an `i < N` guard. Consecutive threads touch
consecutive addresses, so every access is perfectly coalesced. With 12 bytes
of traffic per FLOP the kernel is purely bandwidth-bound — measure GB/s.

## Pitfalls
- Missing bounds check when `N` is not a multiple of the block size.
- For `N > 2^31` use 64-bit indices.

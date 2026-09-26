---
title: Vector Addition
platform: LeetGPU
url: https://leetgpu.com/challenges/vector-addition
difficulty: easy
tags: [elementwise]
status: solved
---

# Vector Addition

**Platform:** LeetGPU · **Difficulty:** easy · [Problem link](https://leetgpu.com/challenges/vector-addition)

## Problem

Given two float vectors `A` and `B` of length `N` (already on the device), write `C[i] = A[i] + B[i]`.

## Approach

One thread per element. Global index `i = blockIdx.x * blockDim.x + threadIdx.x`;
launch `ceil(N / 256)` blocks of 256 threads and guard with `if (i < n)`.

Consecutive threads touch consecutive addresses, so every load and store is
perfectly **coalesced**. The kernel is memory-bound: 2 loads + 1 store per FLOP,
so the only thing that matters is achieving peak DRAM bandwidth.

## Complexity & performance

| Version | Idea | Runtime | GPU |
|---------|------|---------|-----|
| v1      | one thread per element | | |

## Pitfalls

- Forgetting the `i < n` bounds check when `N` is not a multiple of the block size.
- Integer overflow of the index for very large `N` (use `size_t` / `long long` if `N > 2^31`).

## Takeaways

- The canonical "hello world": 1D grid, bounds check, coalesced access.
- Elementwise kernels are bandwidth-bound; measure GB/s, not FLOP/s.

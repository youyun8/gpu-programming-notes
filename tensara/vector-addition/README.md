---
title: Vector Addition
platform: Tensara
url: https://tensara.org/problems/vector-addition
difficulty: easy
tags: [elementwise, vectorized]
status: solved
---

# Vector Addition

**Platform:** Tensara · **Difficulty:** easy · [Problem link](https://tensara.org/problems/vector-addition)

## Problem

`output[i] = input1[i] + input2[i]` for `i < n`. Inputs are device pointers.
Tensara ranks by measured GFLOPS, so the goal is to saturate memory bandwidth.

## Approach

Same idea as the [LeetGPU version](../../leetgpu/001-vector-addition), with two
bandwidth tweaks:

1. **`float4` vectorized loads/stores** – one 128-bit transaction per thread
   instead of four 32-bit ones, fewer instructions per byte moved.
   Requires 16-byte alignment, which `cudaMalloc` guarantees for the base pointer.
2. **Grid-stride loop** – a fixed, occupancy-friendly grid size processes any `n`.

The `n % 4` tail is handled by a scalar loop.

## Complexity & performance

| Version | Idea | Runtime / GFLOPS | GPU |
|---------|------|------------------|-----|
| v1      | one thread per element | | |
| v2      | float4 + grid-stride | | |

## Pitfalls

- `reinterpret_cast<const float4*>` on a misaligned pointer is undefined
  behaviour — only safe here because the base pointers come from `cudaMalloc`.
- The tail elements (`n` not divisible by 4) are easy to forget.

## Takeaways

- Vectorized memory access is a cheap win for any bandwidth-bound kernel.

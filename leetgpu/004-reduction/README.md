---
title: Reduction
platform: LeetGPU
upstream: medium/4_reduction
url: https://leetgpu.com/challenges/reduction
difficulty: medium
tags: [reduction, warp-shuffle, two-pass]
status: solved
---

# Reduction

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/reduction)

## Problem
Sum a float32 array of up to 10⁸ elements into `output[0]`. The reference
sums in float64, and the tolerance is 1e-5.

## Approach
Two-pass reduction (see [tutorial 03](../../tutorials/03-parallel-reduction.md)):
1. ≤1024 blocks walk the array with a grid-stride loop and `float4` loads, each
   thread accumulating in fp32. The block reduces its threads with warp
   shuffles (`__shfl_down_sync`) plus one shared-memory hop between warps and
   writes one **fp64** partial to a `__device__` array.
2. One block adds the partials in fp64 and writes the float result.

Doing the cross-thread part in fp64 costs nothing (it touches ~1000 values)
and keeps the error well inside 1e-5 even when the sum nearly cancels. The
result is deterministic, unlike a `float` `atomicAdd` into one address.

## Pitfalls
- `atomicAdd(float*)` from every block gives a run-to-run varying result and
  more rounding error.
- Grid-stride with a capped grid means the kernel handles any `N`.

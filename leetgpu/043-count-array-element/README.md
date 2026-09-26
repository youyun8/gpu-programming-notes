---
title: Count Array Element
platform: LeetGPU
upstream: medium/43_count_array_element
url: https://leetgpu.com/challenges/count-array-element
difficulty: medium
tags: [reduction, counting, warp-intrinsics]
status: solved
---

# Count Array Element

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/count-array-element)

## Problem
Count elements equal to `K`.

## Approach
`int4` loads, per-thread counts, `__reduce_add_sync` (the sm_80 single-
instruction warp reduction) and one `atomicAdd` per warp.

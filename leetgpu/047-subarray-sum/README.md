---
title: Subarray Sum
platform: LeetGPU
upstream: medium/47_subarray_sum
url: https://leetgpu.com/challenges/subarray-sum
difficulty: medium
tags: [reduction, integer]
status: solved
---

# Subarray Sum

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/subarray-sum)

## Problem
Sum of `input[S..E]` (inclusive).

## Approach
Grid-stride integer reduction over the range with a warp `__reduce_add_sync`
and one atomic per warp. Integer addition is exact, so atomics are fine here.

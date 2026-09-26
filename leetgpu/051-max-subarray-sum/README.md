---
title: Max Subarray Sum
platform: LeetGPU
upstream: medium/51_max_subarray_sum
url: https://leetgpu.com/challenges/max-subarray-sum
difficulty: medium
tags: [scan, prefix-sum]
status: solved
---

# Max Subarray Sum

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/max-subarray-sum)

## Problem
Maximum sum over all windows of length exactly `w` (`N ≤ 50k`).

## Approach
`window(i) = P[i+w] - P[i]` with prefix sums `P`. `N` is small enough for a
single 1024-thread block: it scans the input chunk by chunk (warp shuffle
scan + warp totals + a carry) into a prefix array, then takes a block-wide max
over all windows. This is `O(N)` total, versus `O(N·w)` for a naive
per-window sum.

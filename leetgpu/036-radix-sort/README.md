---
title: Radix Sort
platform: LeetGPU
upstream: hard/36_radix_sort
url: https://leetgpu.com/challenges/radix-sort
difficulty: hard
tags: [sorting, radix-sort, scan, warp-intrinsics]
status: solved
---

# Radix Sort

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/radix-sort)

## Problem
Sort up to 100M `uint32` keys with radix sort.

## Approach
Stable LSD radix sort, 4 passes of 8 bits. Each pass has three steps:
1. **Histogram:** each 2048-key tile counts its digits in shared memory and
   stores the counts **digit-major** (`hist[digit][tile]`).
2. **Scan:** a device-wide exclusive scan of that table gives the global start
   of every `(digit, tile)` pair. Digit-major order puts all tiles' zeros
   first, then all ones, and so on.
3. **Stable scatter:** each tile walks its keys in order, 256 at a time. Within
   a warp, `__match_any_sync` groups lanes with equal digits and
   `popc(peers & lanes_below)` ranks them. Per-warp digit counts are
   prefix-summed across warps, which gives every key a unique slot that
   preserves order.

## Pitfalls
- Stability is what makes LSD correct: each pass must preserve the order of
  the previous one.

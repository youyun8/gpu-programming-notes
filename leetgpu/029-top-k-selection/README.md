---
title: Top K Selection
platform: LeetGPU
upstream: medium/29_top_k_selection
url: https://leetgpu.com/challenges/top-k-selection
difficulty: medium
tags: [selection, radix-select, bitonic-sort]
status: solved
---

# Top K Selection

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/top-k-selection)

## Problem
Write the `k` largest of `N` (≤10⁸) floats in descending order; `k` can be
anything up to `N`.

## Approach
1. **Radix select**: map floats to order-preserving `uint32` keys (flip all bits
   of negatives, set the sign bit of positives). Four passes of 8 bits
   each: a histogram of the current digit over the elements that match the
   prefix found so far picks the digit that contains the k-th largest element.
   After 4 passes we know the exact key `T` of the k-th largest element and how
   many copies of `T` belong to the result. All bookkeeping stays in
   `__device__` memory, so there are no host round-trips.
2. **Gather** every key `> T` into a buffer (atomic cursor) and fill the rest with `T`.
3. **Bitonic sort** the `k` survivors in descending order, in shared memory when
   `k ≤ 2048` (the performance case is `k = 100`), otherwise with global passes.

The total is about 5 streaming passes over the input, independent of `k`.

## Pitfalls
- Duplicates of the threshold value: count them, don't compare with `>=`.
- The order-preserving key transform must treat negative floats specially.

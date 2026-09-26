---
title: Parallel Merge
platform: LeetGPU
upstream: medium/71_parallel_merge
url: https://leetgpu.com/challenges/parallel-merge
difficulty: medium
tags: [merge, merge-path, binary-search]
status: solved
---

# Parallel Merge

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/parallel-merge)

## Problem
Merge two sorted float arrays (up to 50M elements in total).

## Approach
**Merge path.** The first `k` outputs of a merge take `i` elements from A
and `k-i` from B. The "co-rank" `i` is found by binary search along the
cross-diagonal (the smallest `i` with `A[i] > B[k-i-1]`). Each thread co-ranks
the start of its own 8-element output segment, so there is no communication
between threads, and then merges its segment sequentially.

## Pitfalls
- Equal keys: consistently take from A first (`<=`), both in the co-rank
  search and in the merge loop, or segments overlap or leave gaps.

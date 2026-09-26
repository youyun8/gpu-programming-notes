---
title: Segmented Exclusive Prefix Sum
platform: LeetGPU
upstream: medium/70_segmented_prefix_sum
url: https://leetgpu.com/challenges/segmented-exclusive-prefix-sum
difficulty: medium
tags: [scan, segmented-scan]
status: solved
---

# Segmented Exclusive Prefix Sum

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/segmented-exclusive-prefix-sum)

## Problem
Exclusive prefix sum that restarts wherever `flags[i] = 1`.

## Approach
A segmented scan is an ordinary scan over `(flag, sum)` pairs with the
associative operator `(f1,s1) ⊕ (f2,s2) = (f1|f2, f2 ? s2 : s1+s2)`, so the
reduce-then-scan structure of [Prefix Sum](../016-prefix-sum) carries over
unchanged: chunk aggregates, a single-block scan of the aggregates, then a
block-level segmented scan per chunk with a sequential 8-item loop per thread.
Sums are kept in fp64, like the reference.

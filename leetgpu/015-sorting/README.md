---
title: Sorting
platform: LeetGPU
upstream: hard/15_sorting
url: https://leetgpu.com/challenges/sorting
difficulty: hard
tags: [sorting, radix-sort]
status: solved
---

# Sorting

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/sorting)

## Problem
Sort up to 1M floats in place.

## Approach
Map floats to order-preserving `uint32` keys (flip all bits of negatives, set
the sign bit of positives), run a **stable LSD radix sort** (4 × 8-bit
passes, see [Radix Sort](../036-radix-sort)), and map back. It is `O(n)` work
and needs no comparisons.

---
title: Array Sorting
platform: Tensara
upstream: array-sort
url: https://tensara.org/problems/array-sort
difficulty: easy
tags: [sorting, radix-sort]
status: solved
---

# Array Sorting

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/array-sort)

## Problem
Sort int32 values ascending.

## Approach
Flipping the sign bit maps signed order onto unsigned order. A stable LSD
radix sort (4 × 8-bit passes, the same code as the
[LeetGPU radix sort](../../leetgpu/036-radix-sort)) then sorts the keys, and
the sign bit is flipped back.

---
title: 2D Subarray Sum
platform: LeetGPU
upstream: medium/48_2d_subarray_sum
url: https://leetgpu.com/challenges/2d-subarray-sum
difficulty: medium
tags: [reduction, integer]
status: solved
---

# 2D Subarray Sum

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/2d-subarray-sum)

## Problem
Sum over a rectangle of an `N×M` int matrix.

## Approach
Flatten the rectangle into one index space (column fastest, so reads stay
coalesced) and reduce as in the 1-D case.

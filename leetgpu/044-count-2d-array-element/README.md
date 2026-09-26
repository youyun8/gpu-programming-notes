---
title: Count 2D Array Element
platform: LeetGPU
upstream: medium/44_count_2d_array_element
url: https://leetgpu.com/challenges/count-2d-array-element
difficulty: medium
tags: [reduction, counting]
status: solved
---

# Count 2D Array Element

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/count-2d-array-element)

## Problem
Count elements equal to `K` in an `N×M` matrix.

## Approach
The matrix is contiguous, so this is the 1-D counting kernel over `N·M` elements.

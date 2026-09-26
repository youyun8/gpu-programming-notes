---
title: Color Inversion
platform: LeetGPU
upstream: easy/7_color_inversion
url: https://leetgpu.com/challenges/color-inversion
difficulty: easy
tags: [elementwise, vectorized, uint8]
status: solved
---

# Color Inversion

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/color-inversion)

## Problem
Invert R, G, B (`255 - v`) of an RGBA `uint8` image in place; alpha stays.

## Approach
Reinterpret the buffer as `uchar4` so one thread loads and stores a whole
pixel in a single 32-bit transaction instead of four byte accesses.

## Pitfalls
- Don't touch the alpha channel.
- `uchar4` needs 4-byte alignment — guaranteed for `cudaMalloc` buffers.

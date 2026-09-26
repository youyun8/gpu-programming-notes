---
title: Gaussian Blur
platform: LeetGPU
upstream: medium/28_gaussian_blur
url: https://leetgpu.com/challenges/gaussian-blur
difficulty: medium
tags: [convolution, stencil, shared-memory]
status: solved
---

# Gaussian Blur

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/gaussian-blur)

## Problem
"Same" 2D convolution with zero padding, odd kernel up to 21×21.

## Approach
Like [2D Convolution](../010-2d-convolution), but the shared input window starts
`kernel/2` before the tile (the halo). Pixels outside the image are stored as
0, so the inner loop has no bounds checks.

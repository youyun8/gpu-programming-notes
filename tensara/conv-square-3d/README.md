---
title: 3D Square Convolution
platform: Tensara
upstream: conv-square-3d
url: https://tensara.org/problems/conv-square-3d
difficulty: hard
tags: [convolution]
status: solved
---

# 3D Square Convolution

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/conv-square-3d)

## Problem
"Same" 3-D convolution with a cubic kernel (`K ≤ 11`).

## Approach
One thread per output voxel with `threadIdx.x` along the innermost axis
(coalesced); the `K³ ≤ 1331` taps are staged in shared memory and read as
broadcasts.

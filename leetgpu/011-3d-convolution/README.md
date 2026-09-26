---
title: 3D Convolution
platform: LeetGPU
upstream: medium/11_3d_convolution
url: https://leetgpu.com/challenges/3d-convolution
difficulty: medium
tags: [convolution, 3d]
status: solved
---

# 3D Convolution

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/3d-convolution)

## Problem
"Valid" 3D cross-correlation, volume ≤ 256³, kernel ≤ 5×5×5.

## Approach
One thread per output voxel with `threadIdx.x` along columns, so warp reads
are coalesced and adjacent taps hit L1. The ≤125 kernel taps sit in shared
memory (broadcast reads); the depth slice is `blockIdx.z`.

## Pitfalls
- Index math in `size_t`: 256³ = 16.7M elements fits in `int`, but the
  products of intermediate terms can overflow in other variants.

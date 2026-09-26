---
title: 3D Max Pooling
platform: Tensara
upstream: max-pool-3d
url: https://tensara.org/problems/max-pool-3d
difficulty: hard
tags: [pooling]
status: solved
---

# 3D Max Pooling

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/max-pool-3d)

## Problem
`max_pool3d` with padding and dilation.

## Approach
One thread per output, looping over the 3-D dilated window.

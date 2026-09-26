---
title: 3D Average Pooling
platform: Tensara
upstream: avg-pool-3d
url: https://tensara.org/problems/avg-pool-3d
difficulty: hard
tags: [pooling]
status: solved
---

# 3D Average Pooling

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/avg-pool-3d)

## Problem
`avg_pool3d` with padding.

## Approach
One thread per output; the divisor is `k³`.

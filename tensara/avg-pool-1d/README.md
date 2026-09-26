---
title: 1D Average Pooling
platform: Tensara
upstream: avg-pool-1d
url: https://tensara.org/problems/avg-pool-1d
difficulty: easy
tags: [pooling]
status: solved
---

# 1D Average Pooling

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/avg-pool-1d)

## Problem
`avg_pool1d` with padding.

## Approach
One thread per output. `count_include_pad=True` (PyTorch's default), so padded
positions count as zeros and the divisor is always `k`.

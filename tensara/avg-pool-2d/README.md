---
title: 2D Average Pooling
platform: Tensara
upstream: avg-pool-2d
url: https://tensara.org/problems/avg-pool-2d
difficulty: medium
tags: [pooling]
status: solved
---

# 2D Average Pooling

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/avg-pool-2d)

## Problem
`avg_pool2d` with padding.

## Approach
One thread per output; the divisor is `k²` (`count_include_pad`).

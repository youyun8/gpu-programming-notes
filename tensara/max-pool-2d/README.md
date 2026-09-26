---
title: 2D Max Pooling
platform: Tensara
upstream: max-pool-2d
url: https://tensara.org/problems/max-pool-2d
difficulty: medium
tags: [pooling]
status: solved
---

# 2D Max Pooling

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/max-pool-2d)

## Problem
`max_pool2d` with padding and dilation.

## Approach
One thread per output (grid-stride), with dilated windows and skipped padding.

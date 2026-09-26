---
title: 1D Max Pooling
platform: Tensara
upstream: max-pool-1d
url: https://tensara.org/problems/max-pool-1d
difficulty: easy
tags: [pooling]
status: solved
---

# 1D Max Pooling

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/max-pool-1d)

## Problem
`max_pool1d` with padding and dilation.

## Approach
One thread per output. Window element `i` is at `pos·stride − pad + i·dilation`,
and padded positions are skipped (equivalent to −∞ padding).

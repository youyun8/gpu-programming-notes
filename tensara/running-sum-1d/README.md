---
title: 1D Running Sum
platform: Tensara
upstream: running-sum-1d
url: https://tensara.org/problems/running-sum-1d
difficulty: easy
tags: [scan, sliding-window]
status: solved
---

# 1D Running Sum

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/running-sum-1d)

## Problem
Sliding-window sums (a conv1d with a kernel of ones and padding `W/2`).

## Approach
An fp64 inclusive prefix sum `P`, after which every window is `P[hi] − P[lo−1]`.
That is `O(N)` regardless of `W`, and double precision avoids the cancellation
of subtracting two large prefix values.

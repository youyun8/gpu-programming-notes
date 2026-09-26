---
title: Softmax
platform: LeetGPU
upstream: medium/5_softmax
url: https://leetgpu.com/challenges/softmax
difficulty: medium
tags: [softmax, reduction, online-softmax]
status: solved
---

# Softmax

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/softmax)

## Problem
Softmax over a single vector of up to 500k floats, using the max trick.

## Approach
Online softmax: a partial result is a pair `(m, s)` = (running max, Σ e^{x-m}),
and two pairs merge as `m = max(m1, m2)`, `s = s1·e^{m1-m} + s2·e^{m2-m}`.
1. Each block reduces its grid-stride slice to one pair (warp shuffles).
2. One block merges the block pairs into the global `(M, S)`.
3. An elementwise kernel writes `e^{x-M} / S`.

This reads the input twice instead of three times (max pass, sum pass,
normalize pass).

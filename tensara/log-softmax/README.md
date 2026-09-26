---
title: Log Softmax
platform: Tensara
upstream: log-softmax
url: https://tensara.org/problems/log-softmax
difficulty: easy
tags: [softmax]
status: solved
---

# Log Softmax

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/log-softmax)

## Problem
`log_softmax` over the columns of an M×N matrix.

## Approach
Warp per row: online `(max, sum)` with a shuffle merge, then
`x − (max + log sum)`.

---
title: Argmax Over Dimension
platform: Tensara
upstream: argmax
url: https://tensara.org/problems/argmax
difficulty: easy
tags: [reduction, argmax]
status: solved
---

# Argmax Over Dimension

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/argmax)

## Problem
Index of the maximum along one dimension of an arbitrary-rank tensor (first index on ties).

## Approach
View the tensor as `(outer, R, inner)`.
- `inner == 1` (reducing the contiguous axis): one warp per output. Lanes
  stride the row and a shuffle reduction combines `(value, index)` pairs, with
  the smaller index winning ties.
- Otherwise: one thread per output loops over `R` with stride `inner`, which
  stays coalesced across the warp.

The `shape` array can be a host or a device pointer depending on the harness,
so it is read with `cudaMemcpyDefault` (unified addressing works out which).

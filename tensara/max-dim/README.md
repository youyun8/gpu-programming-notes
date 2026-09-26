---
title: Max Over Dimension
platform: Tensara
upstream: max-dim
url: https://tensara.org/problems/max-dim
difficulty: easy
tags: [reduction]
status: solved
---

# Max Over Dimension

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/max-dim)

## Problem
Max along a dimension (keepdim).

## Approach
Generic `(outer, R, inner)` reduction: warp per output when the axis is
contiguous, otherwise thread per output with coalesced strided loops.

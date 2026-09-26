---
title: Sum Over Dimension
platform: Tensara
upstream: sum-dim
url: https://tensara.org/problems/sum-dim
difficulty: easy
tags: [reduction]
status: solved
---

# Sum Over Dimension

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/sum-dim)

## Problem
Sum along a dimension (keepdim).

## Approach
Generic `(outer, R, inner)` reduction with a float accumulator.

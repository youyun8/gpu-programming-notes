---
title: Cumulative Sum
platform: Tensara
upstream: cumsum
url: https://tensara.org/problems/cumsum
difficulty: medium
tags: [scan]
status: solved
---

# Cumulative Sum

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/cumsum)

## Problem
Inclusive prefix sum.

## Approach
Reduce-then-scan over 2048-element chunks with fp64 carries: chunk totals, a
single-block scan of the totals, then block scans of the chunks.

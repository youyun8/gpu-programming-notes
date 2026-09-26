---
title: Image Histogram
platform: Tensara
upstream: histogram
url: https://tensara.org/problems/histogram
difficulty: easy
tags: [histogram, atomics, privatization]
status: solved
---

# Image Histogram

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/histogram)

## Problem
Histogram of `clamp(pixel, 0, bins−1)`, returned as floats.

## Approach
A privatized shared-memory histogram (shared atomics), merged into the zeroed
global histogram once per block, with a global-atomic fallback for very large
bin counts.

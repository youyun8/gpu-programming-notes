---
title: Edge Detection
platform: Tensara
upstream: edge-detect
url: https://tensara.org/problems/edge-detect
difficulty: easy
tags: [stencil, reduction]
status: solved
---

# Edge Detection

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/edge-detect)

## Problem
Central-difference gradient magnitude (zero border), normalized so the maximum becomes 255.

## Approach
Kernel 1 writes the magnitudes and merges the global maximum with `atomicMax`
on the float's bits. This works because non-negative IEEE floats order like
their integer bit patterns. Kernel 2 rescales.

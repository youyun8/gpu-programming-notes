---
title: Box Blur
platform: Tensara
upstream: box-blur
url: https://tensara.org/problems/box-blur
difficulty: easy
tags: [stencil, separable]
status: solved
---

# Box Blur

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/box-blur)

## Problem
Mean over the in-bounds part of a k×k window.

## Approach
A box filter is **separable**: a row pass and then a column pass, `O(k)` per
pixel instead of `O(k²)`. The divisor is the product of the in-bounds window
sizes in each direction.

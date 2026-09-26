---
title: Mean Squared Error Loss
platform: Tensara
upstream: mse-loss
url: https://tensara.org/problems/mse-loss
difficulty: easy
tags: [loss, reduction]
status: solved
---

# Mean Squared Error Loss

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/mse-loss)

## Problem
Mean squared error over a tensor of arbitrary shape.

## Approach
Two-pass reduction with fp64 block partials. The element count is the product
of `shape` (read with `cudaMemcpyDefault`).

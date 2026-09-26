---
title: Mean Squared Error
platform: LeetGPU
upstream: medium/27_mean_squared_error
url: https://leetgpu.com/challenges/mean-squared-error
difficulty: medium
tags: [reduction, two-pass]
status: solved
---

# Mean Squared Error

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/mean-squared-error)

## Problem
`mean((pred - target)²)` over up to 10⁸ elements.

## Approach
Two-pass reduction as in [Reduction](../004-reduction), squaring the
differences on the fly.

---
title: Sigmoid Linear Unit
platform: LeetGPU
upstream: easy/52_silu
url: https://leetgpu.com/challenges/sigmoid-linear-unit
difficulty: easy
tags: [elementwise, activation]
status: solved
---

# Sigmoid Linear Unit

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/sigmoid-linear-unit)

## Problem
`SiLU(x) = x · σ(x)`.

## Approach
Compute `x / (1 + e^{-x})` — one `expf` and one division. For very negative `x`,
`e^{-x}` overflows to `inf` and the result correctly becomes `-0`.

## Pitfalls
- `__expf` is faster but less accurate; with a 1e-5 tolerance keep `expf`.

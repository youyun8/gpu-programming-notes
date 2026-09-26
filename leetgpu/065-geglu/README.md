---
title: Gaussian Error Gated Linear Unit
platform: LeetGPU
upstream: easy/65_geglu
url: https://leetgpu.com/challenges/gaussian-error-gated-linear-unit
difficulty: easy
tags: [elementwise, activation, gated]
status: solved
---

# Gaussian Error Gated Linear Unit

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/gaussian-error-gated-linear-unit)

## Problem
Split into halves; `out = x1 · GELU(x2)` with the exact erf-based GELU.

## Approach
`0.5 · x2 · (1 + erf(x2/√2))` using `erff` — matches the reference within 1e-4.

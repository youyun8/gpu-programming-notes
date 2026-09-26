---
title: Swish-Gated Linear Unit
platform: LeetGPU
upstream: easy/54_swiglu
url: https://leetgpu.com/challenges/swish-gated-linear-unit
difficulty: easy
tags: [elementwise, activation, gated]
status: solved
---

# Swish-Gated Linear Unit

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/swish-gated-linear-unit)

## Problem
Split `x` into halves `x1, x2`; `out = SiLU(x1) · x2` (length `N/2`).

## Approach
One thread per output reads `x[i]` and `x[i + N/2]` — both loads are coalesced.

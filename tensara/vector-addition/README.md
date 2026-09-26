---
title: Vector Addition
platform: Tensara
upstream: vector-addition
url: https://tensara.org/problems/vector-addition
difficulty: easy
tags: [elementwise, vectorized, memory-bound]
status: solved
---

# Vector Addition

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/vector-addition)

## Problem
`C = A + B` for float32 vectors (up to 2³⁰ elements).

## Approach
A grid-stride loop over `float4` (16-byte loads/stores) with a scalar tail.
The grid is capped, so a fixed number of blocks covers any `n`, and vector
access halves the instruction count per byte compared with scalar code.

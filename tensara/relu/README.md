---
title: ReLU
platform: Tensara
upstream: relu
url: https://tensara.org/problems/relu
difficulty: easy
tags: [elementwise, activation]
status: solved
---

# ReLU

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/relu)

## Problem
Elementwise `max(0, x)` over an M×N matrix.

## Approach
The shared Tensara elementwise template: a grid-stride `float4` loop plus a
scalar tail. The matrix shape only matters through `M·N`.

---
title: Leaky ReLU
platform: Tensara
upstream: leaky-relu
url: https://tensara.org/problems/leaky-relu
difficulty: easy
tags: [elementwise, activation]
status: solved
---

# Leaky ReLU

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/leaky-relu)

## Problem
`x > 0 ? x : α·x` elementwise (α is a runtime parameter).

## Approach
Elementwise `float4` template; α is passed by value.

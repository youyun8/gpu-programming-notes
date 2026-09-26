---
title: GELU
platform: Tensara
upstream: gelu
url: https://tensara.org/problems/gelu
difficulty: easy
tags: [elementwise, activation]
status: solved
---

# GELU

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/gelu)

## Problem
GELU with the tanh approximation.

## Approach
`0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))` in the elementwise `float4` template.

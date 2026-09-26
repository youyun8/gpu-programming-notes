---
title: Softplus
platform: Tensara
upstream: soft-plus
url: https://tensara.org/problems/soft-plus
difficulty: easy
tags: [elementwise, activation]
status: solved
---

# Softplus

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/soft-plus)

## Problem
`log(1 + eˣ)` elementwise.

## Approach
Like PyTorch, return `x` for `x > 20` (where `log1p(eˣ)` equals `x` in fp32)
and `log1pf(expf(x))` otherwise.

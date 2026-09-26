---
title: SELU
platform: Tensara
upstream: selu
url: https://tensara.org/problems/selu
difficulty: easy
tags: [elementwise, activation]
status: solved
---

# SELU

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/selu)

## Problem
SELU with the fixed self-normalizing constants `α ≈ 1.6733`, `λ ≈ 1.0507`.

## Approach
`λ·(x > 0 ? x : α·expm1(x))` in the elementwise template.

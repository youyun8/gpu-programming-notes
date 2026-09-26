---
title: ELU
platform: Tensara
upstream: elu
url: https://tensara.org/problems/elu
difficulty: easy
tags: [elementwise, activation]
status: solved
---

# ELU

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/elu)

## Problem
ELU: `x` for positive inputs, `α(eˣ − 1)` otherwise.

## Approach
Elementwise `float4` template. `expm1f` keeps precision near zero, where
`expf(x) − 1` cancels.

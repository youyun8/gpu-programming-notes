---
title: Swish
platform: Tensara
upstream: swish
url: https://tensara.org/problems/swish
difficulty: easy
tags: [elementwise, activation]
status: solved
---

# Swish

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/swish)

## Problem
Swish / SiLU: `x·σ(x)`.

## Approach
`x / (1 + e^{−x})` in the elementwise template.

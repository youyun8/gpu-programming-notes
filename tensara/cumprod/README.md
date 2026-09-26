---
title: Cumulative Product
platform: Tensara
upstream: cumprod
url: https://tensara.org/problems/cumprod
difficulty: medium
tags: [scan]
status: solved
---

# Cumulative Product

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/cumprod)

## Problem
Inclusive prefix product.

## Approach
The same scan pipeline as cumsum with the multiplicative operator (the scan
is generic in the associative operator).

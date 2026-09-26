---
title: L2 Normalization
platform: Tensara
upstream: l2-norm
url: https://tensara.org/problems/l2-norm
difficulty: easy
tags: [normalization, reduction]
status: solved
---

# L2 Normalization

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/l2-norm)

## Problem
Divide each row by its L2 norm (+1e-10).

## Approach
Block per row: block reduction of `x²`, then rescale.

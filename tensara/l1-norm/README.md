---
title: L1 Normalization
platform: Tensara
upstream: l1-norm
url: https://tensara.org/problems/l1-norm
difficulty: easy
tags: [normalization, reduction]
status: solved
---

# L1 Normalization

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/l1-norm)

## Problem
Divide each row by its L1 norm (+1e-10).

## Approach
Block per row: pass 1 is a block reduction of `|x|`, pass 2 rescales the
(cache-resident) row.

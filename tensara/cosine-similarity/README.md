---
title: Cosine Similarity
platform: Tensara
upstream: cosine-similarity
url: https://tensara.org/problems/cosine-similarity
difficulty: easy
tags: [loss, reduction]
status: solved
---

# Cosine Similarity

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/cosine-similarity)

## Problem
`1 − cos(p_i, t_i)` per row.

## Approach
Block per row reduces the three dot products `p·t`, `p·p` and `t·t` in a
single pass; the denominator is clamped like PyTorch's (`ε = 1e-8`).

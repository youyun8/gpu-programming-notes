---
title: Diagonal Matrix Multiplication
platform: Tensara
upstream: diagonal-matmul
url: https://tensara.org/problems/diagonal-matmul
difficulty: easy
tags: [elementwise]
status: solved
---

# Diagonal Matrix Multiplication

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/diagonal-matmul)

## Problem
`diag(a)·B`.

## Approach
This just scales row `i` of `B` by `a[i]`, so it's an elementwise kernel.
Never build the N×N diagonal matrix.

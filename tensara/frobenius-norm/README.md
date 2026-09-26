---
title: Frobenius Normalization
platform: Tensara
upstream: frobenius-norm
url: https://tensara.org/problems/frobenius-norm
difficulty: easy
tags: [normalization, reduction]
status: solved
---

# Frobenius Normalization

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/frobenius-norm)

## Problem
Divide a whole tensor by its Frobenius norm.

## Approach
Three kernels: grid-stride `Σx²` with fp64 block partials, one block for
`1/‖x‖`, then an elementwise scale.

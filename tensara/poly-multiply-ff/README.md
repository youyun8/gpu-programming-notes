---
title: Polynomial Multiplication over Finite Field 
platform: Tensara
upstream: poly-multiply-ff
url: https://tensara.org/problems/poly-multiply-ff
difficulty: medium
tags: [finite-field, convolution]
status: solved
---

# Polynomial Multiplication over Finite Field 

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/poly-multiply-ff)

## Problem
Polynomial product over `F_p`, `p = 2³¹ − 1` (length `2n−1`).

## Approach
`p − 1` has no large power-of-two factor, so there is no direct NTT. For the
tested sizes an `O(n²)` convolution with shared-memory tiles is fast: each
product is reduced with the Mersenne fold, and up to 2⁸ reduced terms are
summed in 64 bits before a final fold.

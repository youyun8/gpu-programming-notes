---
title: Vector Multiplication over Finite Field
platform: Tensara
upstream: vector-multiply-ff
url: https://tensara.org/problems/vector-multiply-ff
difficulty: medium
tags: [finite-field, integer, mersenne]
status: solved
---

# Vector Multiplication over Finite Field

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/vector-multiply-ff)

## Problem
Elementwise `a·b mod (2³¹ − 1)`.

## Approach
Mersenne reduction: since `2³¹ ≡ 1 (mod p)`, a 62-bit product reduces to
`(x & p) + (x >> 31)`. Two folds and one conditional subtract replace the
64-bit division.

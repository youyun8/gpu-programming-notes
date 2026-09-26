---
title: ECC Point Negation (Batched)
platform: Tensara
upstream: ecc-point-negation
url: https://tensara.org/problems/ecc-point-negation
difficulty: easy
tags: [finite-field, integer]
status: solved
---

# ECC Point Negation (Batched)

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/ecc-point-negation)

## Problem
Negate points `(x, y) → (x, −y mod p)` over `p = 2⁶¹ − 1`.

## Approach
Bandwidth-bound: each thread writes both outputs as one 16-byte `ulonglong2`.

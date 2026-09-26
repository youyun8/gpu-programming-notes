---
title: RMS Normalization
platform: Tensara
upstream: rms-norm
url: https://tensara.org/problems/rms-norm
difficulty: easy
tags: [normalization]
status: solved
---

# RMS Normalization

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/rms-norm)

## Problem
`x / sqrt(mean(x²) + 1e-5)` per row, without weights.

## Approach
Block per row: block reduction, then rescale.

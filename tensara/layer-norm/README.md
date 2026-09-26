---
title: Layer Normalization
platform: Tensara
upstream: layer-norm
url: https://tensara.org/problems/layer-norm
difficulty: medium
tags: [normalization]
status: solved
---

# Layer Normalization

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/layer-norm)

## Problem
LayerNorm over the last three dims of `(B, F, D1, D2)` with elementwise γ/β.

## Approach
Each batch element is a contiguous group. One block per group computes the
mean, then the **centered** variance (two fp64 passes, no cancellation), then
writes the output.

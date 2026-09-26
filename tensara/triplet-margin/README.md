---
title: Triplet Margin Loss
platform: Tensara
upstream: triplet-margin
url: https://tensara.org/problems/triplet-margin
difficulty: medium
tags: [loss, reduction]
status: solved
---

# Triplet Margin Loss

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/triplet-margin)

## Problem
Mean triplet-margin loss with `‖a − p + ε‖` distances (`ε = 1e-6`, as in `pairwise_distance`).

## Approach
Block per sample computes both distances in one pass; a final block averages in fp64.

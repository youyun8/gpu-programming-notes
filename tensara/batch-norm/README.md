---
title: Batch Normalization
platform: Tensara
upstream: batch-norm
url: https://tensara.org/problems/batch-norm
difficulty: medium
tags: [normalization]
status: solved
---

# Batch Normalization

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/batch-norm)

## Problem
BatchNorm2d in training mode without affine parameters: per-channel statistics over `(B, D1, D2)`.

## Approach
One block per channel walks the `B` contiguous `D1·D2` chunks of that channel
(coalesced) and reduces the mean and centered variance in fp64 before
normalizing.

---
title: Layer Normalization
platform: LeetGPU
upstream: medium/113_layer_normalization
url: https://leetgpu.com/challenges/layer-normalization
difficulty: medium
tags: [normalization]
status: solved
---

# Layer Normalization

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/layer-normalization)

## Problem
LayerNorm over the feature dimension of `N×C`.

## Approach
Warp per row, three passes over the (L1-resident) row: mean, centered variance,
write.

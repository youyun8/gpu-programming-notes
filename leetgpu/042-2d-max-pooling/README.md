---
title: 2D Max Pooling
platform: LeetGPU
upstream: medium/42_2d_max_pooling
url: https://leetgpu.com/challenges/2d-max-pooling
difficulty: medium
tags: [pooling]
status: solved
---

# 2D Max Pooling

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/2d-max-pooling)

## Problem
`max_pool2d` over an `N×C×H×W` tensor with kernel, stride and padding.

## Approach
One thread per output element with a grid-stride loop. Padded positions are
skipped, which is equivalent to padding with −∞ as PyTorch does.

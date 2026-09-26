---
title: 2D Convolution with ReLU and HardSwish
platform: Tensara
upstream: conv2d-relu-hardswish
url: https://tensara.org/problems/conv2d-relu-hardswish
difficulty: medium
tags: [convolution, fusion]
status: solved
---

# 2D Convolution with ReLU and HardSwish

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/conv2d-relu-hardswish)

## Problem
Same-padded conv2d → ReLU → HardSwish.

## Approach
The banded [conv-2d](../conv-2d) kernel with the activations fused into the store.

---
title: Hard Sigmoid
platform: Tensara
upstream: hard-sigmoid
url: https://tensara.org/problems/hard-sigmoid
difficulty: easy
tags: [elementwise, activation]
status: solved
---

# Hard Sigmoid

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/hard-sigmoid)

## Problem
Hard sigmoid (PyTorch definition): `clamp(x/6 + 1/2, 0, 1)`.

## Approach
Branch-free `fminf/fmaxf` in the elementwise template.

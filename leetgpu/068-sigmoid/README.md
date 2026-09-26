---
title: Sigmoid Activation
platform: LeetGPU
upstream: easy/68_sigmoid
url: https://leetgpu.com/challenges/sigmoid-activation
difficulty: easy
tags: [elementwise, activation, vectorized]
status: solved
---

# Sigmoid Activation

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/sigmoid-activation)

## Problem
`Y = 1 / (1 + e^{-X})`.

## Approach
`float4` elementwise kernel with `expf` (accurate to the 1e-5 tolerance).

---
title: Causal Depthwise Conv1d
platform: LeetGPU
upstream: medium/90_causal_depthwise_conv1d
url: https://leetgpu.com/challenges/causal-depthwise-conv1d
difficulty: medium
tags: [convolution, depthwise, ssm]
status: solved
---

# Causal Depthwise Conv1d

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/causal-depthwise-conv1d)

## Problem
Causal per-channel 1-D convolution (`K ≤ 8`) over a channels-last `(B, L, D)` tensor.

## Approach
`threadIdx.x` runs along channels, so every tap reads a coalesced row. Each
thread keeps its channel's ≤8 weights in registers and computes 8 consecutive
positions, reusing those weights.

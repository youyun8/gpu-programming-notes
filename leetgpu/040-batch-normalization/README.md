---
title: Batch Normalization
platform: LeetGPU
upstream: medium/40_batch_normalization
url: https://leetgpu.com/challenges/batch-normalization
difficulty: medium
tags: [normalization, column-reduction, welford]
status: solved
---

# Batch Normalization

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/batch-normalization)

## Problem
BatchNorm forward over the batch dimension of an `N×C` matrix.

## Approach
- **Statistics:** a 32×8 block owns 32 channels. `threadIdx.x` indexes the
  channel (coalesced row reads) and `threadIdx.y` strides the rows. Each
  thread runs Welford's online mean/M2 in fp64; the 8 partial states are merged
  with Chan's parallel formula.
- **Normalize:** an elementwise kernel computes `γ·(x-μ)·rstd + β`.

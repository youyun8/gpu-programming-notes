---
title: 1D Convolution
platform: Tensara
upstream: conv-1d
url: https://tensara.org/problems/conv-1d
difficulty: easy
tags: [convolution, shared-memory]
status: solved
---

# 1D Convolution

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/conv-1d)

## Problem
"Same" 1-D convolution with odd kernels of up to ~8K taps.

## Approach
Each block produces 1024 outputs (4 per thread). Large kernels are processed
in **2048-tap chunks**: per chunk the block stages the taps and the matching
input window in shared memory, so memory use is bounded for any `K` and every
input element is read from DRAM about once per chunk.

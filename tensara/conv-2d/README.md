---
title: 2D Convolution
platform: Tensara
upstream: conv-2d
url: https://tensara.org/problems/conv-2d
difficulty: medium
tags: [convolution, shared-memory]
status: solved
---

# 2D Convolution

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/conv-2d)

## Problem
"Same" 2-D convolution with odd kernels of up to 127×127.

## Approach
A 32×32 output tile (4 rows per thread). The kernel is walked in **bands of
8 rows**; per band the block stages those kernel rows and the input window
they touch in shared memory. This bounds shared memory (~27 KB) even for
127×127 kernels.

---
title: Linear Recurrence
platform: LeetGPU
upstream: medium/82_linear_recurrence
url: https://leetgpu.com/challenges/linear-recurrence
difficulty: medium
tags: [scan, linear-recurrence, ssm]
status: solved
---

# Linear Recurrence

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/linear-recurrence)

## Problem
`h[t] = a[t]·h[t-1] + x[t]` along sequences of up to 65k.

## Approach
Each step is an affine map `h ↦ a·h + x`, and composing affine maps is
associative: `(A1,X1) then (A2,X2) = (A1·A2, A2·X1 + X2)`, so the recurrence
is a **parallel scan**. One block per sequence: each thread folds its
contiguous chunk into one map, a block scan over the maps (fp64) gives each
thread its incoming `h`, and each thread replays its chunk. This is the same
idea that makes Mamba's selective scan parallel.

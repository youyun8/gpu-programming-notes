---
title: Fused Residual Add and RMS Norm
platform: LeetGPU
upstream: medium/83_fused_residual_add_rms_norm
url: https://leetgpu.com/challenges/fused-residual-add-and-rms-norm
difficulty: medium
tags: [normalization, fusion]
status: solved
---

# Fused Residual Add and RMS Norm

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/fused-residual-add-and-rms-norm)

## Problem
`out = (x + r) / rms(x + r) · w` per row.

## Approach
One block per row. The sum `z = x + r` is never written: pass 1 reduces `Σz²`
with `float4` loads, and pass 2 recomputes `z` (the row is still in L1/L2)
and writes the output. That is one read of each input plus one write, the
minimum possible.

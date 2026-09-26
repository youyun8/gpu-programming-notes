---
title: LoRA Linear
platform: LeetGPU
upstream: medium/85_lora_linear
url: https://leetgpu.com/challenges/lora-linear
difficulty: medium
tags: [gemm, lora, fusion]
status: solved
---

# LoRA Linear

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/lora-linear)

## Problem
`out = x·Wᵀ + s·(x·Aᵀ)·Bᵀ`.

## Approach
1. `hidden = s·x·Aᵀ` (`batch × rank`, tiny).
2. `out = [x | hidden] · [W | B]ᵀ`: a single NT-GEMM whose K loop runs over
   **two concatenated segments** (`d_in`, then `rank`). The base and low-rank
   paths accumulate into the same registers, so the output is written once.

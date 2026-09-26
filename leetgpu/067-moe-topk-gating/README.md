---
title: MoE Top-K Gating
platform: LeetGPU
upstream: medium/67_moe_topk_gating
url: https://leetgpu.com/challenges/moe-top-k-gating
difficulty: medium
tags: [moe, top-k, softmax, warp-intrinsics]
status: solved
---

# MoE Top-K Gating

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/moe-top-k-gating)

## Problem
Per token: pick the top-`k` of `E ≤ 256` expert logits (descending) and
compute the softmax over them.

## Approach
One warp per token. Each lane holds `E/32 ≤ 8` logits in registers. `k` rounds
of a warp arg-max (a `(value, index)` butterfly with lower-index tie-break, as
in `torch.topk`) select the winners; the first winner is the softmax shift,
and the weights are normalized at the end.

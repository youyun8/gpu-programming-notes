---
title: DPO Sequence Loss
platform: LeetGPU
upstream: medium/108_dpo_sequence_loss
url: https://leetgpu.com/challenges/dpo-sequence-loss
difficulty: medium
tags: [reduction, rl]
status: solved
---

# DPO Sequence Loss

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/dpo-sequence-loss)

## Problem
DPO loss `mean(softplus(−z))`.

## Approach
Stable softplus `max(−z,0) + log1p(e^{−|z|})` inside a single-block reduction.

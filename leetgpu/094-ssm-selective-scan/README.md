---
title: SSM Selective Scan
platform: LeetGPU
upstream: medium/94_ssm_selective_scan
url: https://leetgpu.com/challenges/ssm-selective-scan
difficulty: medium
tags: [ssm, mamba, scan]
status: solved
---

# SSM Selective Scan

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/ssm-selective-scan)

## Problem
Mamba selective scan: `h = exp(Δ·A)·h + Δ·B·u`, `y = C·h + skip·u`.

## Approach
Channels are independent. Each thread owns one `(batch, channel)` and keeps
its whole state vector (`d_state ≤ 64`) and its row of `A` in registers
while stepping through time. All threads in a block share the batch index, so
`B[b,t,:]` and `C[b,t,:]` are staged in shared memory 32 timesteps at a time
and read as broadcasts; `u` and `Δ` reads are coalesced across channels.

## Pitfalls
- Loops over the state are fully unrolled with a fixed bound (64) and guarded,
  so the arrays stay in registers instead of spilling to local memory.

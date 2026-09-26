---
title: Prefix Sum
platform: LeetGPU
upstream: medium/16_prefix_sum
url: https://leetgpu.com/challenges/prefix-sum
difficulty: medium
tags: [scan, prefix-sum, reduce-then-scan]
status: solved
---

# Prefix Sum

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/prefix-sum)

## Problem
Inclusive prefix sum of up to 10⁸ floats.

## Approach
Reduce-then-scan with 2048-element chunks (256 threads × 8 items):
1. `blockTotals`: each block sums its chunk.
2. `scanTotals`: one block turns the chunk totals into exclusive offsets,
   walking them 256 at a time with a running carry.
3. `scanChunks`: each block loads its chunk into shared memory (coalesced).
   Each thread scans its 8 consecutive items sequentially, the per-thread
   totals are scanned across the block (warp `__shfl_up_sync` scan + a scan of
   the warp totals), the chunk offset is added, and the chunk is written back
   coalesced.

Offsets are kept in fp64, so rounding does not build up across ~50k chunks.

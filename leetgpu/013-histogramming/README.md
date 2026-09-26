---
title: Histogramming
platform: LeetGPU
upstream: medium/13_histogramming
url: https://leetgpu.com/challenges/histogramming
difficulty: medium
tags: [histogram, atomics, privatization]
status: solved
---

# Histogramming

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/histogramming)

## Problem
Count occurrences of values in `[0, num_bins)` (`num_bins ≤ 1024`) over up to 10⁸ ints.

## Approach
**Privatization:** each block builds its own histogram in shared memory with
shared-memory atomics (much cheaper, and contention stays within the block),
then adds its non-zero bins to the global histogram. Global atomics drop from
`N` to `blocks × bins`.

## Pitfalls
- The output must be zeroed first (`cudaMemset`). The harness gives you an
  uninitialized buffer.

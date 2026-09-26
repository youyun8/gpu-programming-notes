---
title: Softmax Attention
platform: LeetGPU
upstream: medium/6_softmax_attention
url: https://leetgpu.com/challenges/softmax-attention
difficulty: medium
tags: [attention, flash-attention, online-softmax, shared-memory]
status: solved
---

# Softmax Attention

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/softmax-attention)

## Problem
`softmax(Q Kᵀ / √d) V` with `Q: M×d`, `K, V: N×d`, `d ≤ 128`, `M, N ≤ 100k`.

## Approach
A FlashAttention-style single pass that never materializes the `M×N` score matrix:
- One warp per query row, 4 warps per block. The block stages tiles of 32
  keys of `K` and `V` in shared memory; all 4 query rows reuse them.
- **Lane `l` scores key `l`** of the tile (a full `d`-long dot product against
  the query held in shared memory). A tile then needs just one warp-max and
  one warp-sum, instead of a reduction per key.
- Online softmax: when the running max grows, the denominator and the output
  accumulator are rescaled by `e^{m_old - m_new}`.
- Each lane accumulates output columns `lane, lane+32, …`; the tile's
  probabilities are broadcast with `__shfl_sync`.
- The `K` tile uses an odd pitch (129 floats), so the per-lane row reads hit 32
  different banks.

## Pitfalls
- Warps past the last query row still take part in the `__syncthreads()` of
  every tile; they just skip the final store.

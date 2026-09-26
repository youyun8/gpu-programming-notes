---
title: All-Pairs Shortest Paths
platform: LeetGPU
upstream: hard/73_all_pairs_shortest_paths
url: https://leetgpu.com/challenges/all-pairs-shortest-paths
difficulty: hard
tags: [graph, floyd-warshall, blocked-algorithm]
status: solved
---

# All-Pairs Shortest Paths

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/all-pairs-shortest-paths)

## Problem
Floyd–Warshall all-pairs shortest paths (`N ≤ 4096`).

## Approach
**Blocked Floyd–Warshall** with 32×32 tiles. Each round `b` has three phases:
1. The diagonal tile runs its 32 k-steps in shared memory.
2. The row-`b` and column-`b` panel tiles update against it.
3. Every other tile takes `min(d_ij, d_ib + d_bj)` over the round's 32 `k`,
   with both panels staged in shared memory.

Each round touches every tile once, so DRAM traffic is about 32× lower than
with one kernel per `k`. Out-of-range entries are padded with `+∞`.

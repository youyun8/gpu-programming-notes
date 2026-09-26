---
title: BFS Shortest Path
platform: LeetGPU
upstream: hard/46_bfs_shortest_path
url: https://leetgpu.com/challenges/bfs-shortest-path
difficulty: hard
tags: [graph, bfs, persistent-kernel]
status: solved
---

# BFS Shortest Path

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/bfs-shortest-path)

## Problem
Shortest path length on a 4-connected grid with obstacles (`-1` if unreachable).

## Approach
Level-synchronous BFS in **one persistent block**. The level barrier is a
`__syncthreads()` instead of a kernel launch, which matters because a maze
can have hundreds of thousands of levels. The frontier lives in two global
queues, and neighbours are claimed with `atomicCAS` on a visited map, so every
cell is enqueued once. Total work is `O(rows·cols)`.

## Pitfalls
- Multi-block BFS needs a grid-wide barrier per level (cooperative launch) or
  one kernel launch per level. For a single query, one block is the better
  trade-off.

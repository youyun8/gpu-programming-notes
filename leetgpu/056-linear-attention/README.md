---
title: Linear Self-Attention
platform: LeetGPU
upstream: hard/56_linear_attention
url: https://leetgpu.com/challenges/linear-self-attention
difficulty: hard
tags: [attention, linear-attention]
status: solved
---

# Linear Self-Attention

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/linear-self-attention)

## Problem
`φ(Q)(φ(K)ᵀV) / (φ(Q)·Σφ(K))` with `φ = elu + 1`.

## Approach
Associativity turns `O(M²d)` attention into `O(Md²)`:
1. `S = φ(K)ᵀV` (`d×d`) and `z = Σφ(K)`: one thread per entry reduces over all
   rows in fp64 (coalesced `V` reads, broadcast `φ(K)`).
2. One block per query row computes `φ(Q_i)` and its dot with `z` once (in
   shared memory), then thread `j` forms `(φ(Q_i)S)_j`.

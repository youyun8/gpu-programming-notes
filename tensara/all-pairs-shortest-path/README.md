---
title: All-Pairs Shortest Path
platform: Tensara
upstream: all-pairs-shortest-path
url: https://tensara.org/problems/all-pairs-shortest-path
difficulty: medium
tags: [graph, floyd-warshall]
status: solved
---

# All-Pairs Shortest Path

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/all-pairs-shortest-path)

## Problem
All-pairs shortest paths on a weighted adjacency matrix (`0` = no edge, `−1` = unreachable).

## Approach
A prep kernel maps `0 → ∞` and zeroes the diagonal. **Blocked Floyd–Warshall**
(32×32 tiles, three phases per round, shared with the LeetGPU version) does
the work, and a fix-up maps `∞ → −1`.

---
title: Single Source Shortest Path
platform: Tensara
upstream: shortest-path
url: https://tensara.org/problems/shortest-path
difficulty: medium
tags: [graph, bellman-ford]
status: solved
---

# Single Source Shortest Path

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/shortest-path)

## Problem
Single-source shortest paths on a dense graph with positive weights.

## Approach
Bellman–Ford relaxation, parallel over destinations: thread `v` scans column
`v`, which is coalesced across the warp for each `u`. It **stops early** once a
sweep changes nothing, typically after about as many sweeps as the graph's
diameter rather than `N−1`.

---
title: Minimum Spanning Tree
platform: Tensara
upstream: min-spanning-tree
url: https://tensara.org/problems/min-spanning-tree
difficulty: medium
tags: [graph, prim]
status: solved
---

# Minimum Spanning Tree

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/min-spanning-tree)

## Problem
Total weight of a minimum spanning tree (dense adjacency, `0` = no edge).

## Approach
Prim's algorithm in **one block**: each of the `N−1` steps is a block-wide
arg-min followed by a relaxation of every vertex's best edge with the new
vertex's row. It uses only block barriers, with no launch per step. The
total is accumulated in fp64.

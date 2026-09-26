---
title: K-Means Clustering
platform: LeetGPU
upstream: hard/20_kmeans_clustering
url: https://leetgpu.com/challenges/k-means-clustering
difficulty: hard
tags: [clustering, atomics, privatization]
status: solved
---

# K-Means Clustering

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/k-means-clustering)

## Problem
`max_iterations` rounds of Lloyd's k-means on 2-D points (`k ≤ 1000`).

## Approach
Per iteration:
- **Assign:** one thread per point with the centroids in shared memory. The
  squared distance is computed without FMA contraction to match the
  reference's `argmin`, and ties go to the lowest index. Each block accumulates
  per-cluster sums and counts in **shared memory** (fp64 atomics), then flushes
  them with one global atomic per cluster (privatization).
- **Update:** one thread per cluster computes the mean; empty clusters keep
  their centroid.

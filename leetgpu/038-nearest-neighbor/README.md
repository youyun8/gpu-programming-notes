---
title: Nearest Neighbor
platform: LeetGPU
upstream: medium/38_nearest_neighbor
url: https://leetgpu.com/challenges/nearest-neighbor
difficulty: medium
tags: [brute-force, shared-memory, exact-arithmetic]
status: solved
---

# Nearest Neighbor

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/nearest-neighbor)

## Problem
For each of `N` 3-D points, the index of the closest other point (exact match required).

## Approach
Brute force `O(N²)` with shared-memory tiling: each thread owns a query
point, and the block streams all points through shared memory (SoA layout,
broadcast reads).

Because the check is exact, the distance is computed **exactly as the
reference does**: `(dx·dx + dy·dy) + dz·dz` with `__fmul_rn`/`__fadd_rn`, so
the compiler cannot contract it into FMAs, which round differently. Ties go to
the lower index, like `argmin`.

## Pitfalls
- A k-d tree or grid would be asymptotically faster for large `N`, but at the
  benchmark size (10k) the brute force is compute-dense and simple.

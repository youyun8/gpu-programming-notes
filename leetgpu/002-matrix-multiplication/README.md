---
title: Matrix Multiplication
platform: LeetGPU
url: https://leetgpu.com/challenges/matrix-multiplication
difficulty: easy
tags: [gemm, shared-memory, tiling]
status: solved
---

# Matrix Multiplication

**Platform:** LeetGPU · **Difficulty:** easy · [Problem link](https://leetgpu.com/challenges/matrix-multiplication)

## Problem

`A` is `M x N`, `B` is `N x K`, both row-major float32 on the device.
Compute `C = A * B` (`M x K`).

> Verify the dimension naming against the starter code — platforms differ on
> whether the inner dimension is called `N` or `K`.

## Approach

**v1 – naive:** one thread per output element, loops over the inner dimension
reading directly from global memory. Each element of `A` and `B` is read
`K` / `M` times from DRAM (mostly served by L1/L2, but still slow).

**v2 – shared-memory tiling (submitted):** each 16x16 block computes a 16x16
tile of `C`. It walks the inner dimension in 16-wide steps; per step, every
thread loads one element of `A` and one of `B` into shared memory, the block
synchronizes, then each thread accumulates 16 products from shared memory.
Global traffic drops by a factor of `kTile`. See
[tutorials/04-tiled-matmul.md](../../tutorials/04-tiled-matmul.md).

## Complexity & performance

| Version | Idea | Runtime | GPU |
|---------|------|---------|-----|
| v1      | naive, one thread per output | | |
| v2      | 16x16 shared-memory tiles | | |

## Pitfalls

- Out-of-range tiles on the edges must load `0.0f`, not skip the load, or the
  `__syncthreads()` pattern and the accumulation break.
- `__syncthreads()` is needed **both** after loading a tile and before
  overwriting it on the next iteration.
- Map `threadIdx.x` to the column so reads of `B` and writes of `C` are coalesced.

## Takeaways

- Tiling = reuse data from fast memory. Arithmetic intensity grows with tile size.
- Next steps: register tiling (each thread computes several outputs), vectorized
  `float4` loads, double buffering, tensor cores (WMMA / MMA).

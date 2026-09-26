# 04 – Tiled Matrix Multiplication

`C (M x K) = A (M x N) · B (N x K)`: `2·M·N·K` FLOPs over `M·N + N·K + M·K`
elements — high arithmetic intensity, so it can be compute-bound *if* data is
reused from fast memory.

## Naive kernel

One thread per `C[row][col]`, loops over `N`. Each thread reads a full row of
`A` and column of `B` from global memory → intensity ≈ 0.25 FLOP/byte. Memory-bound.

## Shared-memory tiling

Split the inner dimension into tiles of width `T`. Each block:

1. Cooperatively loads a `T x T` tile of `A` and of `B` into shared memory.
2. `__syncthreads()`.
3. Each thread does `T` multiply-adds from shared memory.
4. `__syncthreads()` before the next tile overwrites shared memory.

Global loads are reduced by a factor `T`. Full implementation:
[leetgpu/002-matrix-multiplication](../leetgpu/002-matrix-multiplication/solution.cu).

## Beyond shared memory tiling

| Technique | Why |
|-----------|-----|
| 1D / 2D register tiling (each thread computes 4x4 or 8x8 outputs) | Reuse from registers, fewer shared loads per FMA. |
| `float4` loads, transposed `A` tile in smem | Fewer instructions, conflict-free reads. |
| Double buffering / `cp.async` | Overlap next tile load with current compute. |
| Warp tiling | Match the hardware hierarchy block → warp → thread. |
| Tensor cores (WMMA, `mma.sync`, CuTe / CUTLASS) | 8–16x more FLOPs for fp16/bf16/tf32. |

A good naive → tuned progression is typically 1% → 10% → 50% → 80–90% of cuBLAS.

## Checklist

- `threadIdx.x` ↔ column for coalesced `B` reads and `C` writes.
- Zero-pad edge tiles instead of skipping the load.
- Two `__syncthreads()` per tile iteration.

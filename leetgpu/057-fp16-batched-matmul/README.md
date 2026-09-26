---
title: FP16 Batched Matrix Multiplication
platform: LeetGPU
upstream: medium/57_fp16_batched_matmul
url: https://leetgpu.com/challenges/fp16-batched-matrix-multiplication
difficulty: medium
tags: [gemm, fp16, tensor-cores, wmma, batched]
status: solved
---

# FP16 Batched Matrix Multiplication

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/fp16-batched-matrix-multiplication)

## Problem

Batched half-precision GEMM: $C_b = A_b B_b$ for $b = 0..B-1$, with
$A_b \in \mathrm{fp16}^{M\times K}$, $B_b \in \mathrm{fp16}^{K\times N}$, fp32
accumulation, and an fp16 result ($B \le 128$, $M, N, K \le 1024$; benchmark
$256^3$; tolerance `0.05`). It combines the tensor-core kernel of
[GEMM (fp16)](../022-gemm/) with the batching of
[Batched MatMul](../030-batched-matrix-multiplication/).

## Formulation

$$
C_{b,r,c} = \operatorname{fp16}\!\Bigl(\sum_{k=0}^{K-1} \operatorname{fp32}(A_{b,r,k})\,\operatorname{fp32}(B_{b,k,c})\Bigr)
$$

| Symbol | Meaning |
|---|---|
| $B$ | Batch size (`BATCH`) |
| $M,\ N,\ K$ | Output rows, output columns, inner dimension |
| $A_{b,r,k}$ | fp16 element at offset $bMK + rK + k$ |
| $B_{b,k,c}$ | fp16 element at offset $bKN + kN + c$ |
| $C_{b,r,c}$ | fp16 result at offset $bMN + rN + c$ |
| fp32(·), fp16(·) | Widening conversion and round-to-nearest narrowing |

## Approach

- **Grid** $\lceil N/64\rceil \times \lceil M/64\rceil \times B$. Each block
  offsets its three pointers by the batch index (in `size_t`).
- **Block**: 4 warps, a 64 × 64 output tile. Each warp owns 32 × 32 = 2 × 2
  WMMA 16 × 16 × 16 accumulators (fp32).
- **K loop** in slices of 32: stage zero-padded fp16 tiles
  (`a_s[64][40]`, `b_s[32][72]`), then two `kk` steps of 4 `mma_sync` per
  warp.
- **Epilogue**: accumulators go to a shared fp32 tile, then a bounds-checked
  fp16 store.

The shared pitches satisfy WMMA's `ldm` rule (multiple of 8 halves = 16
bytes), and every fragment pointer is 32-byte aligned (see [GEMM](../022-gemm/)).

## Cost Analysis

$$
W = 2BMNK, \qquad Q \approx 2B\left(MK\frac{N}{64} + KN\frac{M}{64}\right) + 2BMN
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs (executed on tensor cores) |
| $Q$ | Bytes: fp16 operands re-read once per tile row/column, output written once (2 bytes/element) |

At $B = 128$, $256^3$: $W = 4.3$ GFLOP, about 14 µs at 312 TFLOP/s (A100
fp16 dense). Each matrix contributes only 16 blocks, so large batches are
what saturate the SMs.

## Pitfalls

- **Unaligned fragment pointers** (see [GEMM](../022-gemm/)).
- **Result conversion.** Rounding once from fp32 at the end matches the
  reference. Accumulating in fp16 would not.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) (WMMA
emulation) at `0.05`, including non-multiples of 16.

## Related

- [GEMM (fp16)](../022-gemm/), [Batched MatMul (fp32)](../030-batched-matrix-multiplication/).

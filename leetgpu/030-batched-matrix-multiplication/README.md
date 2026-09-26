---
title: Batched Matrix Multiplication
platform: LeetGPU
upstream: medium/30_batched_matrix_multiplication
url: https://leetgpu.com/challenges/batched-matrix-multiplication
difficulty: medium
tags: [gemm, batched, register-blocking, shared-memory]
status: solved
---

# Batched Matrix Multiplication

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/batched-matrix-multiplication)

## Problem

Batched fp32 GEMM: for each of $B$ batch entries, $C_b = A_b B_b$ with
$A_b$ of shape $M \times K$ and $B_b$ of shape $K \times N$, all row-major and
contiguous ($1 \le B \le 128$, $1 \le M, N, K \le 1024$; benchmark
$M = N = K = 256$; tolerance `1e-5`). Batched GEMM appears in attention
(one matmul per head) and in grouped convolutions. The batch is simply a third
grid dimension.

## Formulation

$$
C_{b,r,c} = \sum_{k=0}^{K-1} A_{b,r,k}\, B_{b,k,c}, \qquad 0 \le b < B,\ 0 \le r < M,\ 0 \le c < N
$$

$$
\text{offset}(A_{b,r,k}) = bMK + rK + k, \qquad \text{offset}(B_{b,k,c}) = bKN + kN + c, \qquad \text{offset}(C_{b,r,c}) = bMN + rN + c
$$

| Symbol | Meaning |
|---|---|
| $B$ | batch size (`BATCH`) |
| $M,\ N,\ K$ | rows of $A_b$/$C_b$, columns of $B_b$/$C_b$, inner dimension |
| $b$ | batch index |
| $r,\ c,\ k$ | row, column, inner index |
| $A_{b,r,k}$ etc. | elements, with the contiguous 3-D offsets above |

## Approach

The register-blocked SGEMM from [Matrix Multiplication](../002-matrix-multiplication/)
(64 × 64 block tile, 16-wide K slices, 4 × 4 strided outputs per thread, with
$A$ stored transposed in shared memory) is reused unchanged. The only
additions:

- **grid** = $\lceil N/64\rceil \times \lceil M/64\rceil \times B$, where
  `blockIdx.z` is the batch index;
- at kernel entry the three pointers are offset by $bMK$, $bKN$, $bMN$
  (computed in `size_t`).

Every batch entry is independent, so there is no inter-block communication.
With $M = N = 256$ each matrix gives only 16 blocks. The batch dimension is
what fills the GPU: $16 \cdot B$ blocks in total.

## Cost analysis

$$
W = 2BMNK, \qquad Q \approx 4B\left(MK\frac{N}{64} + KN\frac{M}{64} + MN\right), \qquad I \approx 16 \ \text{FLOP/byte}
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs |
| $Q$ | DRAM bytes with 64 × 64 tiling (inputs re-read once per tile row/column, output written once) |
| $I$ | arithmetic intensity for large $M, N$ |

For $B = 128$ at $256^3$: $W \approx 4.3$ GFLOP. The kernel is compute-bound on
fp32 FMA. For small matrices, the "re-read" terms are L2 hits.

## Pitfalls

- **Batch offset overflow.** $bMK$ can exceed $2^{31}$ for $B = 128$ at
  $1024^2$ ($1.3\times10^8$, which still fits, but it is close). `size_t`
  arithmetic removes the concern.
- **Argument order.** The kernel's `(rows, inner, cols)` are $(M, K, N)$.
  Mixing up $K$ and $N$ only fails on non-square tests.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-5`,
including $B = 1$ and dimensions of 1.

## Related

- [Matrix Multiplication](../002-matrix-multiplication/), [FP16 Batched MatMul](../057-fp16-batched-matmul/).
- Tensara [3D MatMul](../../tensara/matmul-3d/), [4D MatMul](../../tensara/matmul-4d/).

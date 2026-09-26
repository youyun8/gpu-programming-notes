---
title: 4D Tensor-Matrix Multiplication
platform: Tensara
upstream: matmul-4d
url: https://tensara.org/problems/matmul-4d
difficulty: hard
tags: [matmul, sgemm, einsum, reshape]
status: solved
---

# 4D Tensor-Matrix Multiplication

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/matmul-4d)

## Problem

Evaluate `einsum("bijl,lk->bijk", A, B)` for a 4-D tensor $A$ of shape
$B\times I\times J\times L$ and a matrix of shape $L\times K$ (largest:
$16\times256\times512\times256$ times $256\times768$). The check is
`rtol = 2e-4`, `atol = 6e-4`.

## Formulation

$$
C_{bijk} = \sum_{l=0}^{L-1} A_{bijl}\,W_{lk}
$$

| Symbol | Meaning |
|---|---|
| $A$ | input tensor $B\times I\times J\times L$ (row-major, $l$ contiguous) |
| $W$ | the matrix operand, $L\times K$ (called `B` in the signature) |
| $C$ | output tensor $B\times I\times J\times K$ |
| $b, i, j$ | free indices carried through |
| $l$ | contracted index; $k$ output column |

All free indices of $A$ precede the contracted one, so they flatten into
one row index:

$$
\rho = (bI + i)J + j, \qquad C_{\rho k} = \sum_l A_{\rho l}W_{lk}, \qquad 0 \le \rho < BIJ
$$

| Symbol | Meaning |
|---|---|
| $\rho$ | flattened row index |
| $BIJ$ | rows of the flattened GEMM (2 M in the largest case) |

## Approach

One launch of the shared kernel with `rows = b*i*j`, `inner = l`, `cols = k`. The shapes are tall and skinny ($L = 32 \dots 256$), so each block runs only 2–16 K-slices; the load/sync overhead per slice matters more than for square GEMMs.

### The shared SGEMM kernel

All matmul pages on Tensara use the same register-blocked FP32 kernel
(`gemmKernel<kTransB, Epi>`):

1. **Block tile $64\times64$**, 256 threads; each thread owns a $4\times4$
   patch of outputs at rows `ty + 16i`, columns `tx + 16j`. The stride-16
   layout makes every store instruction of a warp hit 16 consecutive
   columns (coalesced), and makes shared-memory reads conflict-free.
2. **K-slices of 16.** Per slice the block copies a $64\times16$ panel of
   $A$ (stored transposed as `a_tile[k][m]`) and a $16\times64$ panel of
   $B$ into shared memory (rows padded by 4 floats), then synchronizes.
3. **Inner product in registers.** For each of the 16 values of $k$, a
   thread loads 4 values of $A$ and 4 of $B$ from shared memory and does
   $4\times4 = 16$ FMAs (an outer product).
4. **Epilogue functor.** The accumulator goes through `epi(v, row, col)`
   before the single store. This is where bias, activations, scaling or
   elementwise multiplies are fused, so the product never makes a round
   trip through DRAM.
5. `kTransB = true` reads $B$ as $N\times K$ ("NT", the `nn.Linear`
   weight layout) and transposes it while staging.

Data reuse at each level of the hierarchy:

$$
I_{\text{L2}} = \frac{2\,T_M T_N T_K}{4\,T_K\,(T_M + T_N)} = \frac{T_M T_N}{2\,(T_M + T_N)} = 16\ \tfrac{\text{flop}}{\text{byte}}, \qquad
I_{\text{smem}} = \frac{2 \cdot r_M r_N}{4\,(r_M + r_N)} = 1\ \tfrac{\text{flop}}{\text{byte}}
$$

| Symbol | Meaning |
|---|---|
| $T_M, T_N, T_K$ | block tile: 64, 64, 16 |
| $r_M, r_N$ | per-thread register tile: 4 × 4 |
| $I_{\text{L2}}$ | flops per byte loaded from L2/DRAM into shared memory |
| $I_{\text{smem}}$ | flops per byte read from shared memory (16 FMAs per 8 loads) |

This reaches roughly 40–60 % of FP32 peak. The next steps are the ones
covered in the [SGEMM tutorial](../../tutorials/04-tiled-matmul.md):
$128\times128$ tiles with $8\times8$ per thread, `float4` shared loads,
double-buffered `cp.async` staging, and finally tensor cores (TF32) where
the tolerance allows.

## Cost analysis

$$
W = 2BIJLK, \qquad Q_{\min} = 4\,(BIJL + LK + BIJK)\ \text{bytes}, \qquad T_{\min} = \max\left(\frac{W}{F},\ \frac{Q_{\min}}{\beta}\right)
$$

| Symbol | Meaning |
|---|---|
| $W$ | floating-point operations (one FMA = 2 flops) |
| $Q_{\min}$ | compulsory DRAM bytes (each operand read once, output written once) |
| $F$ | FP32 peak (tens of TFLOP/s on current GPUs) |
| $\beta$ | DRAM bandwidth |
| $T_{\min}$ | roofline lower bound |

Largest case: $W = 2\cdot 2^{21}\cdot256\cdot768 = 0.82$ TFLOP, while
$Q_{\min} \approx 8.6$ GB. The intensity (~95 flop/byte) is above the FP32
ridge point but not by much, so both compute and bandwidth matter.

## Pitfalls

- **Name clash**: the signature's `B` is the matrix; the batch size is
  `b`.
- **Argument order** `(A, B, C, b, i, j, l, k)`: $l$ before $k$.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Matmul 3D](../matmul-3d/), [Matrix Multiplication](../matrix-multiplication/).

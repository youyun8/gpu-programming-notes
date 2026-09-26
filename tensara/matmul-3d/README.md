---
title: 3D Tensor-Matrix Multiplication
platform: Tensara
upstream: matmul-3d
url: https://tensara.org/problems/matmul-3d
difficulty: hard
tags: [matmul, sgemm, batched, reshape]
status: solved
---

# 3D Tensor-Matrix Multiplication

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/matmul-3d)

## Problem

Multiply a 3-D tensor $A$ of shape $N\times M\times K$ by a matrix $B$ of
shape $K\times L$, giving $N\times M\times L$. The tests are big (for
example $64\times4096\times4096$ times $4096\times8192$, 8.8 TFLOP). The
check is `rtol = 2e-4`, `atol = 3e-3`.

## Formulation

$$
C_{bil} = \sum_{k=0}^{K-1} A_{bik}\,B_{kl}
$$

| Symbol | Meaning |
|---|---|
| $A$ | input tensor, $N\times M\times K$, row-major |
| $B$ | shared matrix, $K\times L$ |
| $C$ | output tensor, $N\times M\times L$ |
| $b, i$ | batch and row indices; $k$ reduction; $l$ output column |

Because $B$ is the same for every batch and $A$'s leading two axes are
contiguous, the row index $\rho = bM + i$ turns the problem into one GEMM:

$$
C_{\rho l} = \sum_{k} A_{\rho k}\,B_{kl}, \qquad 0 \le \rho < NM
$$

| Symbol | Meaning |
|---|---|
| $\rho$ | flattened row index over batch and rows |
| $NM$ | number of rows of the flattened GEMM |

## Approach

A single launch of the shared kernel with `rows = n*m`, `inner = k`, `cols = l`. No batching loop is needed, and the larger grid fills the GPU better than $N$ separate GEMMs.

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
W = 2NMKL, \qquad Q_{\min} = 4\,(NMK + KL + NML)\ \text{bytes}, \qquad T_{\min} = \max\left(\frac{W}{F},\ \frac{Q_{\min}}{\beta}\right)
$$

| Symbol | Meaning |
|---|---|
| $W$ | floating-point operations (one FMA = 2 flops) |
| $Q_{\min}$ | compulsory DRAM bytes (each operand read once, output written once) |
| $F$ | FP32 peak (tens of TFLOP/s on current GPUs) |
| $\beta$ | DRAM bandwidth |
| $T_{\min}$ | roofline lower bound |

The largest case is $2\cdot64\cdot4096\cdot4096\cdot8192 = 1.76\times10^{13}$
flop, several seconds of FP32 work at 50 % of peak.

## Pitfalls

- **Reshape, don't batch**: a batched GEMM with $N$ copies of $B$ wastes
  nothing but launches; the reshape is simpler and faster.
- **Row count** $NM$ fits in `int` for the tests; offsets are `size_t`.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Matmul 4D](../matmul-4d/), [Matrix Multiplication](../matrix-multiplication/),
  LeetGPU [Batched Matrix Multiplication](../../leetgpu/030-batched-matrix-multiplication/).

---
title: Matrix Multiplication with Swish and Scaling
platform: Tensara
upstream: matmul-swish-scaling
url: https://tensara.org/problems/matmul-swish-scaling
difficulty: medium
tags: [matmul, sgemm, fusion]
status: solved
---

# Matrix Multiplication with Swish and Scaling

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/matmul-swish-scaling)

## Problem

Compute $O = \text{scale}\cdot\operatorname{swish}(AB)$ for $A$ of size
$M\times K$ and $B$ of size $K\times N$ (sizes 512 … 1024). The check is
`rtol = 5e-4`, `atol = 2e-4`.

## Visual Overview

![O = scale · swish(AB): a plain GEMM with a Swish-and-scale epilogue](figure.svg)

The GEMM tile stays in registers; the epilogue box applies the Swish
activation and the scale before the only write.

## Formulation

$$
G_{ij} = \sum_{k=0}^{K-1} A_{ik}B_{kj}, \qquad O_{ij} = \text{scale}\cdot G_{ij}\,\sigma(G_{ij}), \qquad \sigma(t) = \frac{1}{1 + e^{-t}}
$$

| Symbol | Meaning |
|---|---|
| $A, B$ | row-major GEMM operands |
| $G$ | Product (registers only) |
| $\sigma$ | Logistic sigmoid |
| Scale | Scalar multiplier |
| $O$ | Output, $M\times N$ |

## Approach

The shared kernel ("NN" layout) with a Swish-and-scale epilogue. The only difference from [MatMul + Swish](../matmul-swish/) is the absence of bias and the non-transposed $B$.

### The Shared SGEMM Kernel

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
| $T_M, T_N, T_K$ | Block tile: 64, 64, 16 |
| $r_M, r_N$ | per-thread register tile: 4 × 4 |
| $I_{\text{L2}}$ | Flops per byte loaded from L2/DRAM into shared memory |
| $I_{\text{smem}}$ | Flops per byte read from shared memory (16 FMAs per 8 loads) |

This reaches roughly 40–60 % of FP32 peak. The next steps are the ones
covered in the [SGEMM tutorial](../../tutorials/04-tiled-matmul.md):
$128\times128$ tiles with $8\times8$ per thread, `float4` shared loads,
double-buffered `cp.async` staging, and finally tensor cores (TF32) where
the tolerance allows.

## Cost Analysis

$$
W = 2MNK, \qquad Q_{\min} = 4\,(MK + KN + MN)\ \text{bytes}, \qquad T_{\min} = \max\left(\frac{W}{F},\ \frac{Q_{\min}}{\beta}\right)
$$

| Symbol | Meaning |
|---|---|
| $W$ | floating-point operations (one FMA = 2 flops) |
| $Q_{\min}$ | Compulsory DRAM bytes (each operand read once, output written once) |
| $F$ | FP32 peak (tens of TFLOP/s on current GPUs) |
| $\beta$ | DRAM bandwidth |
| $T_{\min}$ | Roofline lower bound |

At $1024^3$: 2.1 GFLOP on 256 blocks. As for the other small fused GEMMs,
filling the machine matters more than the inner loop.

## Pitfalls

- **Swish of the product**, then scale: $\text{scale}\cdot\operatorname{swish}(G)$,
  not $\operatorname{swish}(\text{scale}\cdot G)$.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [MatMul + Swish](../matmul-swish/), [GEMM × LeakyReLU](../gemm-multiply-leakyrelu/), [Swish](../swish/).

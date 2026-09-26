---
title: GEMM with Bias and ReLU
platform: Tensara
upstream: gemm-relu
url: https://tensara.org/problems/gemm-relu
difficulty: medium
tags: [matmul, sgemm, fusion, linear-layer]
status: solved
---

# GEMM with Bias and ReLU

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/gemm-relu)

## Problem

A fully connected layer with ReLU: $C = \operatorname{ReLU}(AW^{\mathsf T} + \mathbf{b})$
with $A$ of size $B\times N$ (batch × input features), $W$ of size
$M\times N$ (PyTorch `nn.Linear` layout) and bias $\mathbf{b}$ of length $M$.
Sizes: $B = 512 \dots 1024$, $N$ up to 8192, $M$ up to 2048. The check is
`rtol = 3e-3`, `atol = 2e-4`.

## Formulation

$$
Z_{rc} = \sum_{n=0}^{N-1} A_{rn}\,W_{cn} + b_c, \qquad C_{rc} = \max(Z_{rc}, 0)
$$

| Symbol | Meaning |
|---|---|
| $A$ | input activations, $B\times N$ |
| $W$ | weights, $M\times N$: row $c$ holds output feature $c$'s weights |
| $\mathbf{b}$ | bias, length $M$ |
| $Z$ | pre-activation, $B\times M$ (never stored) |
| $C$ | output, $B\times M$ |
| $r, c, n$ | batch row, output feature, input feature |

$W_{cn}$ indexed by $(c, n)$ means the product is $AW^{\mathsf T}$: both
operands are read along their contiguous $n$ axis. In BLAS terms this is
an "NT" GEMM.

## Approach

The shared kernel with `kTransB = true` (it reads $W$ row by row and transposes it into the shared tile) and the epilogue `BiasReluEpi`: `fmaxf(v + bias[c], 0)`. The bias load is a per-column value, served from L1.

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
W = 2BNM, \qquad Q_{\min} = 4\,(BN + MN + BM)\ \text{bytes}, \qquad T_{\min} = \max\left(\frac{W}{F},\ \frac{Q_{\min}}{\beta}\right)
$$

| Symbol | Meaning |
|---|---|
| $W$ | floating-point operations (one FMA = 2 flops) |
| $Q_{\min}$ | compulsory DRAM bytes (each operand read once, output written once) |
| $F$ | FP32 peak (tens of TFLOP/s on current GPUs) |
| $\beta$ | DRAM bandwidth |
| $T_{\min}$ | roofline lower bound |

Unfused (GEMM, then bias, then ReLU) would add two extra read/write passes
over $B\times M$ floats, i.e. $4\cdot 4BM$ bytes; at $1024\times2048$ that is
34 MB, about 17 µs, compared with ~0.4 ms for the GEMM itself.

## Pitfalls

- **Layout of $W$**: using $W$ as $N\times M$ reads the wrong elements;
  the test does not use square $W$ everywhere, so it fails loudly.
- **Bias per output column** $c$, not per row.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [GEMM × LeakyReLU](../gemm-multiply-leakyrelu/), [MatMul + Swish](../matmul-swish/),
  [ReLU](../relu/), LeetGPU [GEMM](../../leetgpu/022-gemm/).

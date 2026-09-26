---
title: Matrix Multiplication with Sigmoid and Sum
platform: Tensara
upstream: matmul-sigmoid-sum
url: https://tensara.org/problems/matmul-sigmoid-sum
difficulty: medium
tags: [matmul, sgemm, fusion, reduction, atomics]
status: solved
---

# Matrix Multiplication with Sigmoid and Sum

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/matmul-sigmoid-sum)

## Problem

Return the single scalar $\sum_{i,j}\sigma\bigl((AB)_{ij}\bigr)$ for $A$ of
size $M\times K$ and $B$ of size $K\times N$ (sizes 512 … 1024). The check
is loose: `rtol = 5e-2`, `atol = 1e-2`.

## Formulation

$$
G_{ij} = \sum_{k=0}^{K-1} A_{ik}B_{kj}, \qquad
\text{result} = \sum_{i=0}^{M-1}\sum_{j=0}^{N-1} \sigma(G_{ij})
$$

| Symbol | Meaning |
|---|---|
| $A, B$ | GEMM operands |
| $G$ | product (never stored) |
| $\sigma$ | logistic sigmoid |
| result | scalar output |

The sum is split by output tile:

$$
\text{result} = \sum_{t} S_t, \qquad S_t = \sum_{(i,j)\in t} \sigma(G_{ij})
$$

| Symbol | Meaning |
|---|---|
| $t$ | a $64\times64$ output tile (one thread block) |
| $S_t$ | the tile's partial sum |

## Approach

A variant of the shared SGEMM whose epilogue does not store anything:

1. Each thread applies $\sigma$ to its 16 accumulators and adds them.
2. The block reduces the 256 per-thread values: a float warp-shuffle
   butterfly, then thread 0 adds the 8 warp sums in `double`.
3. One thread per block does `atomicAdd(&g_sum, S_t)` on a `double`
   (native on sm_60+).
4. A one-thread kernel converts `g_sum` to float into `output` (after a
   reset kernel zeroes it at the start of the call).

The $M\times N$ intermediate never exists; the only global writes are one
atomic per block.

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
| $T_M, T_N, T_K$ | block tile: 64, 64, 16 |
| $r_M, r_N$ | per-thread register tile: 4 × 4 |
| $I_{\text{L2}}$ | flops per byte loaded from L2/DRAM into shared memory |
| $I_{\text{smem}}$ | flops per byte read from shared memory (16 FMAs per 8 loads) |

This reaches roughly 40–60 % of FP32 peak. The next steps are the ones
covered in the [SGEMM tutorial](../../tutorials/04-tiled-matmul.md):
$128\times128$ tiles with $8\times8$ per thread, `float4` shared loads,
double-buffered `cp.async` staging, and finally tensor cores (TF32) where
the tolerance allows.

## Cost Analysis

$$
W = 2MNK, \qquad Q_{\min} = 4\,(MK + KN)\ \text{bytes}, \qquad T_{\min} = \max\left(\frac{W}{F},\ \frac{Q_{\min}}{\beta}\right)
$$

| Symbol | Meaning |
|---|---|
| $W$ | floating-point operations (one FMA = 2 flops) |
| $Q_{\min}$ | compulsory DRAM bytes (each operand read once, output written once) |
| $F$ | FP32 peak (tens of TFLOP/s on current GPUs) |
| $\beta$ | DRAM bandwidth |
| $T_{\min}$ | roofline lower bound |

The output is a single float, so $Q_{\min}$ drops the $MN$ term: at
$1024^3$ the unfused pipeline would write and re-read 4 MB twice, which is
comparable to the GEMM's own input traffic at this size.

## Pitfalls

- **Non-determinism**: the order of the atomic additions varies between
  runs, so the last bits of the double sum vary; after rounding to float
  this is invisible.
- **Reset the accumulator** every call; a `__device__` global keeps its
  value across launches.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [MatMul + Swish](../matmul-swish/), [Sigmoid](../sigmoid/),
  [Frobenius Norm](../frobenius-norm/) (the same two-level reduction).

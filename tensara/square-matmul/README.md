---
title: Square Matrix Multiplication
platform: Tensara
upstream: square-matmul
url: https://tensara.org/problems/square-matmul
difficulty: medium
tags: [matmul, sgemm, register-blocking]
status: solved
---

# Square Matrix Multiplication

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/square-matmul)

## Problem

FP32 product of two $N\times N$ matrices, $N$ from 4096 to 9216. The
check is `rtol = 2e-4`, `atol = 5e-3`. It is [Matrix Multiplication](../matrix-multiplication/)
with $M = N = K$; $N = 6144, 7168, 9216$ are multiples of 64, so no
partial tiles occur.

## Formulation

$$
C_{ij} = \sum_{k=0}^{N-1} A_{ik}\,B_{kj}
$$

| Symbol | Meaning |
|---|---|
| $A, B$ | inputs, $N\times N$, row-major |
| $C$ | output $AB$ |
| $N$ | matrix side |

The accumulated rounding error of a length-$N$ dot product is bounded by

$$
\bigl\lvert \hat{C}_{ij} - C_{ij} \bigr\rvert \le \gamma_N \sum_k \lvert A_{ik}\rvert\,\lvert B_{kj}\rvert, \qquad \gamma_N = \frac{N u}{1 - N u}
$$

| Symbol | Meaning |
|---|---|
| $\hat{C}_{ij}$ | the computed (rounded) value |
| $u$ | unit roundoff, $2^{-24}$ for fp32 |
| $\gamma_N$ | worst-case growth factor (the typical error grows like $\sqrt{N}u$) |

This is why `atol = 5e-3` is needed at $N = 9216$ even though the kernel is
correct: a different summation order than cuBLAS changes the last bits.

## Approach

The shared kernel with `M = N = K = n` and `NoEpi`.

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
W = 2N^3, \qquad Q_{\min} = 4\,(3N^2)\ \text{bytes}, \qquad T_{\min} = \max\left(\frac{W}{F},\ \frac{Q_{\min}}{\beta}\right)
$$

| Symbol | Meaning |
|---|---|
| $W$ | floating-point operations (one FMA = 2 flops) |
| $Q_{\min}$ | compulsory DRAM bytes (each operand read once, output written once) |
| $F$ | FP32 peak (tens of TFLOP/s on current GPUs) |
| $\beta$ | DRAM bandwidth |
| $T_{\min}$ | roofline lower bound |

At $N = 9216$: $W = 1.57$ TFLOP, compute-bound by a wide margin.

## Pitfalls

- **Only one size argument**: `solution(a, b, c, n)`.
- **Rounding**: see the error bound above; do not tighten the kernel's
  accumulation order to "match" cuBLAS, it cannot.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Matrix Multiplication](../matrix-multiplication/), [Symmetric Matmul](../symmetric-matmul/),
  [Lower Triangular Matmul](../lower-trig-matmul/).

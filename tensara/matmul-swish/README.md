---
title: Matrix Multiplication with Swish Activation
platform: Tensara
upstream: matmul-swish
url: https://tensara.org/problems/matmul-swish
difficulty: medium
tags: [matmul, sgemm, fusion, linear-layer]
status: solved
---

# Matrix Multiplication with Swish Activation

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/matmul-swish)

## Problem

Linear layer followed by Swish and a scale:
$\text{out} = s\cdot\operatorname{swish}(xW^{\mathsf T} + \mathbf{b})$ with
$x$ of size $B\times\text{in}$ and $W$ of size $\text{out}\times\text{in}$
(for example $B = 128$, in = 1024, out = 512, $s = 2$). The check is
`rtol = 3e-4`, `atol = 1e-5`.

## Formulation

$$
z_{rc} = \sum_{n=0}^{\text{in}-1} x_{rn}W_{cn} + b_c, \qquad
\text{out}_{rc} = s\,z_{rc}\,\sigma(z_{rc}) = \frac{s\,z_{rc}}{1 + e^{-z_{rc}}}
$$

| Symbol | Meaning |
|---|---|
| $x$ | input, $B\times\text{in}$ |
| $W$ | weights, $\text{out}\times\text{in}$ (`nn.Linear` layout, so the GEMM is "NT") |
| $\mathbf{b}$ | bias, length out |
| $z$ | linear output (kept in registers) |
| $\sigma$ | logistic sigmoid |
| $s$ | `scaling_factor` |
| out | result, $B\times\text{out}$ |

## Approach

The shared kernel with `kTransB = true` and an epilogue that adds the bias, applies $z\,\sigma(z)$ and multiplies by $s$.

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
W = 2B\cdot\text{in}\cdot\text{out}, \qquad Q_{\min} = 4\,(B\cdot\text{in} + \text{out}\cdot\text{in} + B\cdot\text{out})\ \text{bytes}, \qquad T_{\min} = \max\left(\frac{W}{F},\ \frac{Q_{\min}}{\beta}\right)
$$

| Symbol | Meaning |
|---|---|
| $W$ | floating-point operations (one FMA = 2 flops) |
| $Q_{\min}$ | compulsory DRAM bytes (each operand read once, output written once) |
| $F$ | FP32 peak (tens of TFLOP/s on current GPUs) |
| $\beta$ | DRAM bandwidth |
| $T_{\min}$ | roofline lower bound |

At $128\times1024\times512$: $W = 134$ MFLOP, microseconds of work. The grid
is only $8\times2 = 16$ blocks of $64\times64$, far fewer than the SM count:
this size is latency-bound, and split-K (several blocks per output tile
along the reduction) is the fix.

## Pitfalls

- **Tight `atol = 1e-5`**: outputs are small for small $z$; the order
  bias → swish → scale must match the reference.
- **`const float scaling_factor`** comes before the output pointer.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [MatMul + Swish + Scaling](../matmul-swish-scaling/), [Swish](../swish/),
  [GEMM + ReLU](../gemm-relu/), LeetGPU [SwiGLU MLP Block](../../leetgpu/084-swiglu-mlp-block/).

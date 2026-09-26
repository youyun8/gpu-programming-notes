---
title: GEMM with Element-wise Multiply and LeakyReLU
platform: Tensara
upstream: gemm-multiply-leakyrelu
url: https://tensara.org/problems/gemm-multiply-leakyrelu
difficulty: medium
tags: [matmul, sgemm, fusion, activation]
status: solved
---

# GEMM with Element-wise Multiply and LeakyReLU

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/gemm-multiply-leakyrelu)

## Problem

Compute $O = \operatorname{LeakyReLU}_\alpha\bigl((AB)\odot C\bigr)$ for
$A$ of size $M\times K$, $B$ of size $K\times N$ and an elementwise
multiplier $C$ of size $M\times N$ (sizes 512 … 1024). The check is
`rtol = 3e-4`, `atol = 1e-4`.

## Formulation

$$
G_{ij} = \sum_{k=0}^{K-1} A_{ik}B_{kj}, \qquad
H_{ij} = G_{ij}\,C_{ij}, \qquad
O_{ij} = \begin{cases} H_{ij}, & H_{ij} \ge 0 \\ \alpha H_{ij}, & H_{ij} < 0 \end{cases}
$$

| Symbol | Meaning |
|---|---|
| $A, B$ | GEMM operands, row-major |
| $G$ | the product $AB$ (kept in registers) |
| $C$ | elementwise multiplier, $M\times N$ |
| $\odot$ | Hadamard (elementwise) product |
| $H$ | $G\odot C$ |
| $\alpha$ | LeakyReLU slope |
| $O$ | output, $M\times N$ |

## Approach

The shared kernel with the epilogue `MulLeakyEpi{C, ld, alpha}`: it loads $C_{ij}$ at the same coalesced position as the store, multiplies, and applies the slope. $C$ is read exactly once and $G$, $H$ are never written.

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
W = 2MNK, \qquad Q_{\min} = 4\,(MK + KN + 2MN)\ \text{bytes}, \qquad T_{\min} = \max\left(\frac{W}{F},\ \frac{Q_{\min}}{\beta}\right)
$$

| Symbol | Meaning |
|---|---|
| $W$ | floating-point operations (one FMA = 2 flops) |
| $Q_{\min}$ | compulsory DRAM bytes (each operand read once, output written once) |
| $F$ | FP32 peak (tens of TFLOP/s on current GPUs) |
| $\beta$ | DRAM bandwidth |
| $T_{\min}$ | roofline lower bound |

The matrices are small ($1024^3$ is 2.1 GFLOP), so a $64\times64$ tile gives
$16\times16 = 256$ blocks, about 2 per SM; latency and tail effects are
visible, and a split-K or smaller tile would fill the GPU better.

## Pitfalls

- **Argument order**: `(A, B, C, alpha, output, M, N, K)`; `alpha` sits
  between the inputs and the output.
- **`>=` vs `>`**: at $H = 0$ both give 0, so either is fine.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [GEMM + ReLU](../gemm-relu/), [Leaky ReLU](../leaky-relu/),
  [MatMul + Swish + Scaling](../matmul-swish-scaling/).

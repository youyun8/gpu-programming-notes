---
title: 2D Convolution
platform: LeetGPU
upstream: medium/10_2d_convolution
url: https://leetgpu.com/challenges/2d-convolution
difficulty: medium
tags: [convolution, shared-memory, tiling, dynamic-shared-memory]
status: solved
---

# 2D Convolution

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/2d-convolution)

## Problem

"Valid" 2-D cross-correlation of an $R \times C$ float32 image with a
$K_r \times K_c$ kernel ($1 \le R, C \le 3072$, $1 \le K_r, K_c \le 31$;
benchmark $3072 \times 3072$ with a $15\times15$ kernel). The output is
$(R - K_r + 1) \times (C - K_c + 1)$ and the tolerance is `1e-5`. This is the
2-D version of the halo-tiling idea from [1D Convolution](../009-1d-convolution/),
and the basic operation behind blurs, edge detectors and CNN layers.

## Formulation

$$
Y_{ij} = \sum_{m=0}^{K_r-1}\sum_{n=0}^{K_c-1} X_{i+m,\ j+n}\ w_{mn},
\qquad 0 \le i < R - K_r + 1,\ \ 0 \le j < C - K_c + 1
$$

| Symbol | Meaning |
|---|---|
| $R,\ C$ | input rows and columns |
| $K_r,\ K_c$ | kernel rows and columns |
| $X_{ab}$ | input pixel at row $a$, column $b$ (row-major, offset $aC + b$) |
| $w_{mn}$ | kernel weight at row $m$, column $n$ |
| $Y_{ij}$ | output pixel; the output has $C - K_c + 1$ columns |
| $i,\ j$ | output row and column |
| $m,\ n$ | kernel row and column offsets |

### Tile and Halo

A block computes a $32 \times 32$ output tile whose top-left corner is
$(i_0, j_0)$. It needs the input window

$$
X[\,i_0 : i_0 + 32 + K_r - 1,\ \ j_0 : j_0 + 32 + K_c - 1\,]
$$

| Symbol | Meaning |
|---|---|
| $(i_0, j_0)$ | tile origin: $(32\,\texttt{blockIdx.y},\ 32\,\texttt{blockIdx.x})$ |
| $32 + K - 1$ | tile plus halo; at most $62 \times 62$ for $K = 31$ |

## Approach

- **Block shape.** 32 × 8 threads for a 32 × 32 tile: each thread computes 4
  outputs in the same column, at rows $t_y, t_y+8, t_y+16, t_y+24$.
- **Stage.** Copy the kernel ($K_rK_c$ floats) and the window into dynamic
  shared memory: $(K_rK_c + (32+K_r-1)(32+K_c-1))\cdot 4$ bytes, i.e. 19 KB
  at the maximum size. Then `__syncthreads()`.
- **Compute.** Loop over $(m, n)$. The weight is a broadcast read and is used
  for 4 FMAs. The input reads `s_input[(ty + 8r + m)·win_cols + tx + n]`
  are consecutive across the 32 threads of a warp (varying `tx`), so there
  are no bank conflicts.
- **Store** with a bounds check against the output shape.

Each thread keeps 4 accumulators in registers, so the loads of
`s_kernel[m][n]` are amortised over 4 outputs.

## Cost Analysis

$$
W = 2K_rK_c\,(R-K_r+1)(C-K_c+1), \qquad
\text{reuse} = \frac{32\cdot 32\cdot K_rK_c}{(32+K_r-1)(32+K_c-1)}
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs (multiply-add per tap per output) |
| reuse | average number of times each staged input value is used from shared memory |

For the benchmark ($3072^2$ image, $15\times15$ kernel), $W \approx 4.1$
GFLOP. The reuse factor is $1024 \cdot 225 / 46^2 \approx 109$, so DRAM
traffic (≈ 38 MB for the input plus 36 MB of output) is not the bottleneck.
The inner loop is shared-load + FMA bound. The next step up would be register
tiling along the row as well, keeping a sliding window of input values in
registers.

## Pitfalls

- **Separate bounds for window and output.** Window loads check the *input*
  shape; stores check the *output* shape.
- **Kernel size varies per test.** The window pitch `win_cols` is a runtime
  value, so the shared array is dynamic and indexed manually.
- **Large kernels vs. occupancy.** 19 KB per block limits residency to a few
  blocks per SM, which is fine at this scale.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md), including
1 × 1 kernels, kernels as large as the input, and non-square shapes.

## Related

- [1D Convolution](../009-1d-convolution/), [3D Convolution](../011-3d-convolution/),
  [Gaussian Blur](../028-gaussian-blur/), [Jacobi Stencil](../069-jacobi-stencil-2d/).
- Tensara [2D Convolution](../../tensara/conv-2d/), [Box Blur](../../tensara/box-blur/).

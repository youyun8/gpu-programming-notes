---
title: 3D Convolution
platform: LeetGPU
upstream: medium/11_3d_convolution
url: https://leetgpu.com/challenges/3d-convolution
difficulty: medium
tags: [convolution, 3d, shared-memory]
status: solved
---

# 3D Convolution

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/3d-convolution)

## Problem

"Valid" 3-D cross-correlation of a $D \times R \times C$ float32 volume with a
$K_d \times K_r \times K_c$ kernel ($1 \le D, R, C \le 256$,
$1 \le K_d, K_r, K_c \le 5$; tolerance `1e-5`). The output has shape
$(D-K_d+1) \times (R-K_r+1) \times (C-K_c+1)$. This is the basic operation of
video and volumetric (medical-imaging) CNNs. With at most $5^3 = 125$ taps,
the kernel is small and the input fits comfortably in the cache hierarchy.

## Formulation

$$
Y_{z,r,c} = \sum_{a=0}^{K_d-1}\ \sum_{b=0}^{K_r-1}\ \sum_{e=0}^{K_c-1} X_{z+a,\ r+b,\ c+e}\ w_{a,b,e}
$$

$$
\text{offset}_X(z, r, c) = (zR + r)\,C + c, \qquad \text{offset}_Y(z, r, c) = \bigl(z(R-K_r+1) + r\bigr)(C-K_c+1) + c
$$

| Symbol | Meaning |
|---|---|
| $D,\ R,\ C$ | Input depth, rows, columns |
| $K_d,\ K_r,\ K_c$ | Kernel depth, rows, columns |
| $X_{z,r,c}$ | Input voxel (depth slice $z$, row $r$, column $c$) |
| $w_{a,b,e}$ | Kernel tap |
| $Y_{z,r,c}$ | Output voxel |
| $a,\ b,\ e$ | Kernel offsets along depth, rows, columns |
| $\text{offset}$ | Linear index in the row-major (slice, row, column) layout |

## Approach

- **Grid.** $\lceil C_o/32\rceil \times \lceil R_o/8\rceil \times D_o$ blocks
  of $32 \times 8$ threads, one thread per output voxel. `threadIdx.x` runs
  along columns and `blockIdx.z` is the output depth slice.
- **Kernel in shared memory.** The ≤ 125 taps are copied once per block. In
  the inner loop all threads of a warp read the same tap, which is a
  broadcast.
- **Input through L1.** For a fixed tap $(a, b, e)$, a warp reads 32
  consecutive floats of one input row, which is coalesced. Over the $K_c$ taps
  of a row the warp's window slides by one element, so almost every load after
  the first hits in L1. Explicit shared-memory halo tiling (as in
  [2D Convolution](../010-2d-convolution/)) is therefore not needed for
  kernels this small.
- **Early exit after the barrier.** Threads outside the output return only
  *after* the kernel-staging `__syncthreads()`.

## Cost Analysis

$$
W = 2K_dK_rK_c\ D_oR_oC_o, \qquad Q_{\min} = 4\,(DRC + D_oR_oC_o), \qquad I_{\max} = \frac{W}{Q_{\min}} \approx \frac{K_dK_rK_c}{4}
$$

| Symbol | Meaning |
|---|---|
| $D_o,\ R_o,\ C_o$ | Output dimensions $D-K_d+1$, $R-K_r+1$, $C-K_c+1$ |
| $W$ | FLOPs |
| $Q_{\min}$ | Compulsory DRAM traffic: read the input once, write the output once |
| $I_{\max}$ | best-case arithmetic intensity (when caches provide all reuse) |

For a $5^3$ kernel, $I_{\max} \approx 31$ FLOP/byte, so the kernel is compute-
or L1-bound rather than DRAM-bound. Each FMA issues one L1 load plus one
shared load. Register blocking along columns (a thread computes several
consecutive $c$ and slides a window of registers) would cut the L1 loads per
FMA by up to $K_c\times$.

## Pitfalls

- **Layout order.** The flat layout is (slice, row, column). Transposing the
  meaning of $D$ and $R$ still passes on cubes but fails on non-cubic volumes.
- **Return before the barrier.** Out-of-range threads must still help stage the
  kernel and reach `__syncthreads()` before exiting.
- **Tap count.** The static shared array holds 125 taps, which is exactly the
  $5^3$ limit.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md), including
kernels as large as the volume and 1 × 1 × 1 kernels.

## Related

- [1D](../009-1d-convolution/) and [2D Convolution](../010-2d-convolution/).
- Tensara [3D Convolution (square)](../../tensara/conv-square-3d/).

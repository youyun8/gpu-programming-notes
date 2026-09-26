---
title: Gaussian Blur
platform: LeetGPU
upstream: medium/28_gaussian_blur
url: https://leetgpu.com/challenges/gaussian-blur
difficulty: medium
tags: [convolution, stencil, shared-memory, zero-padding]
status: solved
---

# Gaussian Blur

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/gaussian-blur)

## Problem

Blur an $R \times C$ float32 image with a normalised $K_r \times K_c$
Gaussian kernel ($1 \le R, C \le 4096$; $K_r, K_c$ odd in $[3, 21]$;
benchmark $512 \times 512$ with $7\times7$; tolerance `1e-5`). This is a
**"same"** convolution: the output has the input's size, and pixels outside
the image count as 0 (zero padding). The reference is `F.conv2d` with
padding $(K_r/2, K_c/2)$.

## Formulation

$$
Y_{ij} = \sum_{m=0}^{K_r-1}\sum_{n=0}^{K_c-1} \tilde X_{\,i + m - h_r,\ j + n - h_c}\ w_{mn}, \qquad
\tilde X_{ab} = \begin{cases} X_{ab}, & 0 \le a < R,\ 0 \le b < C \\ 0, & \text{otherwise} \end{cases}
$$

| Symbol | Meaning |
|---|---|
| $R,\ C$ | Image rows and columns (also the output size) |
| $K_r,\ K_c$ | Kernel height and width (odd) |
| $h_r,\ h_c$ | Half sizes $\lfloor K_r/2\rfloor$, $\lfloor K_c/2\rfloor$: the kernel's centre offset |
| $w_{mn}$ | Kernel weight, $w_{mn} \ge 0$, $\sum w_{mn} = 1$ |
| $X_{ab}$ | Input pixel |
| $\tilde X_{ab}$ | zero-padded input |
| $Y_{ij}$ | Output pixel, $0 \le i < R$, $0 \le j < C$ |

A true Gaussian kernel is **separable**, $w_{mn} = g_m g_n$. That would allow
two 1-D passes costing $K_r + K_c$ instead of $K_rK_c$ taps per pixel. The
problem passes an arbitrary normalised kernel, so the 2-D form is computed.

## Approach

This is the [2D Convolution](../010-2d-convolution/) kernel with two changes:

1. **Halo offset.** The staged window starts at
   $(i_0 - h_r,\ j_0 - h_c)$, i.e. half a kernel before the $32\times32$
   output tile.
2. **Zero padding at staging time.** While copying the
   $(32 + K_r - 1)\times(32 + K_c - 1)$ window to shared memory, positions
   outside the image are written as 0. The inner FMA loop then needs **no
   boundary checks at all**. Every thread runs the same instruction stream,
   and the edge tiles cost the same as the interior ones.

Each 32 × 8 block computes 4 outputs per thread. Kernel weights are
shared-memory broadcasts, and input reads are consecutive across a warp.

## Cost Analysis

$$
W = 2K_rK_c\,RC, \qquad Q \approx 4RC\left(\frac{(32+K_r-1)(32+K_c-1)}{32\cdot32} + 1\right)
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs |
| $Q$ | DRAM bytes: each tile loads its haloed window, and each output is written once |

Benchmark ($512^2$, $7\times7$): $W \approx 25.7$ MFLOP and $Q \approx 2.4$ MB.
This is a microsecond-scale kernel dominated by launch overhead. At $4096^2$
with $21\times21$ the same code does 14.8 GFLOP from shared memory with a
reuse factor of about $1024\cdot441/52^2 \approx 167$.

## Pitfalls

- **Valid vs. same.** Unlike [2D Convolution](../010-2d-convolution/), the
  output size is the input size, and the tile origin is offset by the half
  kernel.
- **Negative coordinates.** The window can start at row/column $-h$, so the
  bounds check must include `r >= 0 && c >= 0`.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-5`,
including 1-pixel images and $21 \times 21$ kernels larger than the image.

## Related

- [2D Convolution](../010-2d-convolution/), [Jacobi Stencil](../069-jacobi-stencil-2d/).
- Tensara [Box Blur](../../tensara/box-blur/), [Edge Detection](../../tensara/edge-detect/).

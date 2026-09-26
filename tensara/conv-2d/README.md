---
title: 2D Convolution
platform: Tensara
upstream: conv-2d
url: https://tensara.org/problems/conv-2d
difficulty: medium
tags: [convolution, shared-memory, tiling]
status: solved
---

# 2D Convolution

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/conv-2d)

## Problem

"Same" 2-D cross-correlation of an $H\times W$ float32 image with an odd
$K_h\times K_w$ kernel and zero padding. The tests range from
$16384^2$ images with $13\times13$ kernels to $4096^2$ images with
$127\times127$ kernels, so shared-memory use must be bounded for any
kernel size. The check is `rtol = 2e-4`, `atol = 1e-3`.

## Formulation

$$
C[i, j] = \sum_{u=0}^{K_h-1} \sum_{v=0}^{K_w-1} \tilde{A}\bigl[i + u - p_h,\ j + v - p_w\bigr]\; B[u, v], \qquad
p_h = \frac{K_h - 1}{2},\quad p_w = \frac{K_w - 1}{2}
$$

| Symbol | Meaning |
|---|---|
| $A$ | input image $H\times W$, row-major; $\tilde{A}$ is $A$ with zeros outside |
| $B$ | kernel $K_h\times K_w$, both odd |
| $p_h, p_w$ | vertical and horizontal padding (kernel radii) |
| $C$ | output image $H\times W$ |
| $i, j$ | output row and column |
| $u, v$ | kernel row and column |

For an output tile of $T_y\times T_x$ pixels and a band of $b$ kernel rows,
the input window that the band touches is

$$
(T_y + b - 1) \times (T_x + K_w - 1)
$$

| Symbol | Meaning |
|---|---|
| $T_y, T_x$ | output tile height and width (32 × 32) |
| $b$ | kernel rows processed per pass (8) |

## Approach

1. **Output tile $32\times32$**, block $32\times8$ threads, 4 output rows
   per thread held in registers.
2. **Bands of 8 kernel rows.** For each band, the block stages the band's
   kernel rows ($8\times K_w$) and its input window
   ($39\times(31 + K_w)$, zero-padded) in shared memory. The window for
   $K_w = 127$ is $39\times158$ floats, so shared memory stays under 29 KB
   for every test case.
3. **Inner loop**: for each tap $(u, v)$, one broadcast load of the weight
   and four shared loads feed four FMAs. `threadIdx.x` indexes columns, so
   the 32 lanes read 32 consecutive words (conflict-free).
4. The kernel is a template over an **epilogue functor**, which lets
   [Conv2D + ReLU + HardSwish](../conv2d-relu-hardswish/) reuse it.

## Cost analysis

$$
W = 2HWK_hK_w, \qquad
Q \approx 4HW\left(1 + \left\lceil \frac{K_h}{8} \right\rceil \frac{39\,(31+K_w)}{32\cdot 32}\right)\ \text{bytes}
$$

| Symbol | Meaning |
|---|---|
| $W$ | floating-point operations |
| $Q$ | bytes through L2: one staged window per band per tile (the ratio is window area over tile area), plus the output write |

At $4096^2$ with $127^2$ taps, $W = 541$ GFLOP: seconds of FP32 work, so
arithmetic dominates everything. At $16384^2$ with $13^2$ taps, $W = 91$
GFLOP against 2 GB of compulsory traffic, which is still compute-bound. As
in 1-D, each FMA costs one shared load; register tiling along $x$ and
separable or FFT methods (for $K \ge 63$) are the big levers.

## Pitfalls

- **Shared memory for large kernels**: staging the whole window for a
  $127\times127$ kernel needs $158^2\cdot 4 = 100$ KB per block; banding
  keeps it bounded.
- **Zero-fill** the window when it crosses the image edge; do not clamp.
- **Two `__syncthreads`** per band: before overwriting the tiles and after
  loading them.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Conv 1D](../conv-1d/), [Conv2D + ReLU + HardSwish](../conv2d-relu-hardswish/),
  [Box Blur](../box-blur/), LeetGPU [2D Convolution](../../leetgpu/010-2d-convolution/).

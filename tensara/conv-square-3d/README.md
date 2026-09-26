---
title: 3D Square Convolution
platform: Tensara
upstream: conv-square-3d
url: https://tensara.org/problems/conv-square-3d
difficulty: hard
tags: [convolution, 3d, shared-memory]
status: solved
---

# 3D Square Convolution

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/conv-square-3d)

## Problem

"Same" 3-D cross-correlation of an $n\times n\times n$ float32 volume with
a $K\times K\times K$ kernel ($K$ odd, $3 \le K \le 11$) and zero padding
$K/2$. Volumes range from $32^3$ to $512^3$. The check is `rtol = 1e-3`,
`atol = 1e-2`.

## Formulation

$$
C[z, y, x] = \sum_{a=0}^{K-1}\sum_{b=0}^{K-1}\sum_{c=0}^{K-1}
\tilde{A}\bigl[z + a - p,\ y + b - p,\ x + c - p\bigr]\; B[a, b, c], \qquad p = \frac{K-1}{2}
$$

| Symbol | Meaning |
|---|---|
| $A$ | Input volume, $n^3$, row-major ($x$ contiguous); $\tilde{A}$ is zero outside |
| $B$ | Cubic kernel, $K^3$ taps |
| $p$ | Padding on every face |
| $C$ | Output volume, $n^3$ |
| $z, y, x$ | Output coordinates (depth, row, column) |
| $a, b, c$ | Kernel offsets |

## Approach

1. **Grid** $(n/32,\ n/8,\ n)$ of $32\times8$ blocks: `blockIdx.z` is the
   output plane, `threadIdx.x` the contiguous $x$ axis.
2. **Kernel in shared memory**: up to $11^3 = 1331$ taps (5.3 KB). All
   lanes of a warp read the same tap at the same time, which is a
   broadcast.
3. **Input straight from global memory**: neighbouring lanes read
   neighbouring $x$, so each tap row is a coalesced 128-byte access, and
   the $K^2$ rows touched by a warp are reused by the other warps of the
   block and by the next plane's block through L1/L2.
4. Out-of-range planes and rows are skipped with `continue`, so the border
   costs nothing.

## Cost Analysis

$$
W = 2n^3K^3, \qquad Q_{\min} = 8n^3\ \text{bytes}, \qquad I = \frac{W}{Q_{\min}} = \frac{K^3}{4}
$$

| Symbol | Meaning |
|---|---|
| $W$ | floating-point operations |
| $Q_{\min}$ | Compulsory DRAM bytes (read $A$ once, write $C$ once) |
| $I$ | best-case arithmetic intensity, flops per byte |

For $512^3$ with $K = 9$: $W = 196$ GFLOP and $I = 182$ flop/B, which is
compute-bound. Because the input is not staged in shared memory, each FMA
also needs one L1 load. A version with a shared input halo tile
($(8 + K - 1)\times(32 + K - 1)$ per plane) plus register reuse along $z$
is the next step.

## Pitfalls

- **`size` is the edge length**, not the element count.
- **Tolerances are loose** (`atol = 1e-2`) because up to 1331 products are
  summed in a different order from cuDNN.
- The grid's $z$ dimension is limited to 65535, which is fine for
  $n \le 512$.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Conv 2D](../conv-2d/), [Avg Pool 3D](../avg-pool-3d/),
  LeetGPU [3D Convolution](../../leetgpu/011-3d-convolution/).

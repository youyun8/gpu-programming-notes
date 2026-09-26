---
title: 2D Max Pooling
platform: LeetGPU
upstream: medium/42_2d_max_pooling
url: https://leetgpu.com/challenges/2d-max-pooling
difficulty: medium
tags: [pooling, cnn, stencil]
status: solved
---

# 2D Max Pooling

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/2d-max-pooling)

## Problem

2-D max pooling over an $N \times C \times H \times W$ float32 tensor (NCHW)
with a square window of size $k$, stride $s$ and zero-width padding $p$
($N \le 100$, $C \le 512$, $H, W \le 1024$, $k, s, p \le 16$; benchmark
$N = 4$, $k = 3$, $s = 2$; tolerance `1e-5`). It matches
`F.max_pool2d(…, kernel_size=k, stride=s, padding=p)`.

## Formulation

$$
H_o = \left\lfloor\frac{H + 2p - k}{s}\right\rfloor + 1, \qquad W_o = \left\lfloor\frac{W + 2p - k}{s}\right\rfloor + 1
$$

$$
Y_{n,c,y,x} = \max_{\substack{0 \le a, b < k \\ 0 \le ys - p + a < H \\ 0 \le xs - p + b < W}} X_{n,\,c,\ ys - p + a,\ xs - p + b}
$$

| Symbol | Meaning |
|---|---|
| $N,\ C,\ H,\ W$ | Batch, channels, input height and width |
| $k$ | Window size (`kernel_size`) |
| $s$ | Stride |
| $p$ | Padding on each side (padded cells never win: treated as $-\infty$) |
| $H_o,\ W_o$ | Output height and width |
| $X_{n,c,h,w}$ | Input element, offset $((nC + c)H + h)W + w$ |
| $Y_{n,c,y,x}$ | Output element, offset $((nC + c)H_o + y)W_o + x$ |
| $a,\ b$ | Offsets inside the window |

PyTorch requires $p \le k/2$, so every window contains at least one real
input cell and the max is always finite.

## Approach

- One thread per output element, flattened over $N \cdot C \cdot H_o \cdot W_o$,
  with a grid-stride loop (the grid is capped at 65 535 blocks).
- Decode $(\text{plane}, y, x)$ from the flat index, with $x$ fastest. A warp
  then writes consecutive outputs, and its input reads are strided by $s$
  within the same rows.
- Loop over the $k \times k$ window. Rows or columns outside the image are
  `continue`d, which is exactly padding with $-\infty$.

With $k \le 16$, the windows of neighbouring outputs overlap when $s < k$
(e.g. $3\times3$, stride 2). Those repeated reads are served by L1/L2, and
explicit shared-memory tiling brings little for such small windows.

## Cost Analysis

$$
W_{\text{cmp}} = k^2 N C H_o W_o, \qquad Q_{\min} = 4NC\,(HW + H_oW_o)
$$

| Symbol | Meaning |
|---|---|
| $W_{\text{cmp}}$ | Comparisons (`fmaxf`) |
| $Q_{\min}$ | Compulsory bytes: read the input once, write the output once |

Pooling is memory-bound: at most $k^2/4$ comparisons per input byte in the
worst case, usually far fewer.

## Pitfalls

- **Output size formula.** Use floor division, as PyTorch does with
  `ceil_mode=False`.
- **Padding value.** Padding with 0 instead of $-\infty$ gives wrong results
  for all-negative windows at the border.
- **Large tensors.** $N C H W$ can exceed $2^{31}$ (100 × 512 × 1024² is
  $5\times10^{10}$), so all flat indices are `size_t`.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md), including
$k = 1$, $s > k$ (gaps between windows) and $p = k/2$.

## Related

- Tensara [Max Pool 1D](../../tensara/max-pool-1d/), [2D](../../tensara/max-pool-2d/),
  [3D](../../tensara/max-pool-3d/), [Avg Pool 2D](../../tensara/avg-pool-2d/).

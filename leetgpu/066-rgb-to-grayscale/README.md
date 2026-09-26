---
title: RGB to Grayscale
platform: LeetGPU
upstream: easy/66_rgb_to_grayscale
url: https://leetgpu.com/challenges/rgb-to-grayscale
difficulty: easy
tags: [image, elementwise, strided-access]
status: solved
---

# RGB to Grayscale

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/rgb-to-grayscale)

## Problem

Convert an $H\times W$ RGB image (float32, interleaved R, G, B per pixel,
values in $[0, 255]$) to grayscale with the ITU-R BT.601 luma weights
($WH \le 4.2$M; benchmark $2048 \times 2048$; tolerance `1e-5`).

## Formulation

$$
Y_p = 0.299\,R_p + 0.587\,G_p + 0.114\,B_p, \qquad (R_p, G_p, B_p) = (x_{3p},\ x_{3p+1},\ x_{3p+2})
$$

| Symbol | Meaning |
|---|---|
| $p$ | pixel index $yW + x$, $0 \le p < WH$ |
| $x_k$ | input array (length $3WH$) |
| $R_p,\ G_p,\ B_p$ | red, green and blue values of pixel $p$ |
| $Y_p$ | output luma |
| 0.299, 0.587, 0.114 | BT.601 weights (they sum to 1; green dominates because the eye is most sensitive to it) |

## Approach

One thread per pixel, reading 3 consecutive floats at `input + 3p` and
writing one float.

### Is the Strided Read Wasteful?

Per instruction, a warp reads 32 floats with a stride of 12 bytes: a 384-byte
span that is only one-third used. But the three instructions (R, G, B)
together cover exactly that same 384-byte span. The span is fetched from
DRAM once, and the second and third loads hit in L1. DRAM traffic therefore
equals the input size. The inefficiency is only in the number of L1
transactions, not in bandwidth.

Alternatives: stage the pixels through shared memory with coalesced
`float4` loads, or let each thread handle 4 pixels = 3 `float4` loads.
These reduce instruction count, but not DRAM bytes.

## Cost Analysis

$$
Q = 12WH + 4WH = 16WH\ \text{bytes}, \qquad W_{\text{flop}} = 5WH
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes: read 3 floats and write 1 per pixel |
| $W_{\text{flop}}$ | 3 multiplies + 2 adds per pixel |

Benchmark ($2048^2$): 67 MB, i.e. ≈ 34 µs at 2 TB/s.

## Pitfalls

- **Operation order.** PyTorch computes `0.299*R + 0.587*G + 0.114*B` left to
  right in float32. The compiler may contract into FMAs, which changes the
  last bit, but that is well within `1e-5` for values ≤ 255.
- **Double literals.** Writing `0.299` instead of `0.299f` silently promotes to
  float64 arithmetic.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-5`.

## Related

- [Color Inversion](../007-color-inversion/), Tensara [Grayscale](../../tensara/grayscale/).

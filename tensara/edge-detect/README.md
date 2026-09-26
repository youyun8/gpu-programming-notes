---
title: Edge Detection
platform: Tensara
upstream: edge-detect
url: https://tensara.org/problems/edge-detect
difficulty: easy
tags: [stencil, reduction, atomics, image-processing]
status: solved
---

# Edge Detection

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/edge-detect)

## Problem

Gradient-magnitude edge detection on an $h\times w$ float32 image
($1024\times768$ … $4096\times4096$): central differences in $x$ and $y$,
magnitude, border pixels set to 0, then the whole image is rescaled so its
maximum becomes 255. The check is `rtol = atol = 1e-3`.

## Formulation

$$
G_x[i, j] = \frac{I[i, j+1] - I[i, j-1]}{2}, \qquad
G_y[i, j] = \frac{I[i+1, j] - I[i-1, j]}{2}, \qquad
M[i, j] = \sqrt{G_x[i, j]^2 + G_y[i, j]^2}
$$

for interior pixels $1 \le i \le h-2$, $1 \le j \le w-2$, and $M = 0$ on the
1-pixel border. Then

$$
\text{out}[i, j] = \begin{cases} 255\,\dfrac{M[i, j]}{M_{\max}}, & M_{\max} > 0 \\ M[i, j], & \text{otherwise} \end{cases}, \qquad
M_{\max} = \max_{i, j} M[i, j]
$$

| Symbol | Meaning |
|---|---|
| $I$ | Input image, $h\times w$, row-major |
| $G_x, G_y$ | Horizontal and vertical central differences |
| $M$ | Gradient magnitude, $\ge 0$ |
| $M_{\max}$ | Global maximum of $M$ |
| out | Output, scaled to $[0, 255]$ |

The global maximum is found with an integer `atomicMax` on the raw float
bits. This is valid because, for non-negative IEEE-754 numbers,

$$
0 \le a < b \iff \operatorname{bits}(a) < \operatorname{bits}(b)
$$

| Symbol | Meaning |
|---|---|
| $\operatorname{bits}(a)$ | The 32-bit pattern of float $a$ read as an unsigned integer (`__float_as_uint`) |

## Approach

1. **`resetMax`** sets the device global `g_max_bits` to 0 (a one-thread
   launch, so each call starts fresh).
2. **`magnitude`**: one thread per pixel (grid-stride). The four
   neighbours are $\pm1$ and $\pm w$ from the pixel; a warp touches three
   consecutive row segments, all served by L1. Each thread tracks its
   local maximum, the warp reduces it with `__shfl_xor_sync`, and one
   lane per warp does `atomicMax` on the bits.
3. **`normalize`** reads `g_max_bits` once and rescales in place (skipped
   when the image is flat).

## Cost Analysis

$$
Q \approx 4hw\ (\text{read } I) + 4hw\ (\text{write } M) + 8hw\ (\text{rescale}) = 16hw\ \text{bytes}, \qquad
\#\text{atomics} = \frac{\#\text{threads}}{32}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes |
| #atomics | One per warp (at most 32 K here) |

At $4096^2$: 268 MB, ~0.13 ms at 2 TB/s. The second pass costs half the
traffic; it is unavoidable unless the maximum were known in advance (for
example, recomputing $M$ in the second pass instead of storing it trades
the $M$ round trip for re-reading $I$, which is the same amount).

## Pitfalls

- **Border pixels are 0**, including in the maximum.
- **`atomicMax` on float bits** only works because all values are $\ge 0$;
  negative floats order in reverse.
- **Flat image**: if $M_{\max} = 0$, do not divide.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Box Blur](../box-blur/), [Grayscale](../grayscale/), [Conv 2D](../conv-2d/),
  LeetGPU [Jacobi Stencil 2D](../../leetgpu/069-jacobi-stencil-2d/).

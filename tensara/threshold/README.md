---
title: Image Thresholding
platform: Tensara
upstream: threshold
url: https://tensara.org/problems/threshold
difficulty: easy
tags: [elementwise, image-processing, float4]
status: solved
---

# Image Thresholding

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/threshold)

## Problem

Binary thresholding of a grayscale float32 image of height $h$ and width
$w$ ($1024\times768$ to $3840\times2160$) with a runtime threshold $\theta$
(64, 128 or 192): pixels strictly above $\theta$ become 255, all others 0.
The checker compares exactly.

## Formulation

$$
\text{out}[i, j] = \begin{cases} 255, & I[i, j] > \theta \\ 0, & \text{otherwise} \end{cases}
$$

| Symbol | Meaning |
|---|---|
| $I$ | input image, $h\times w$ float32, values in $[0, 255]$ |
| $\theta$ | threshold (`threshold_value`) |
| out | binary output image, values in $\{0, 255\}$ |

## Approach

All Tensara elementwise problems share one kernel shape:

1. **`float4` grid-stride loop.** The buffer is viewed as $\lfloor n/4 \rfloor$
   16-byte vectors; each iteration loads one `float4`, applies the scalar
   function to the four lanes, and stores one `float4`. `cudaMalloc`
   returns 256-byte-aligned pointers, so the reinterpretation is safe.
2. **Scalar tail** for the last $n \bmod 4$ elements.
3. **Launch** 256-thread blocks, capped at 4096 blocks; the grid-stride
   loop covers any size, and 4096 × 256 threads are enough to saturate DRAM.
4. The function is a `__forceinline__` device function, so the loop body is
   branch-free apart from the select in the function itself.

The select `x > threshold ? 255.0f : 0.0f` has no arithmetic at all; it is a pure bandwidth test.

## Cost analysis

$$
n = hw, \qquad Q = 8\,n\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $n$ | number of elements |
| $Q$ | compulsory DRAM traffic: read the input(s) once, write the output once |
| $\beta$ | DRAM bandwidth (about 2–3 TB/s on current data-centre GPUs) |
| $T_{\min}$ | bandwidth lower bound on the kernel time |

For $3840\times2160$: $Q = 66$ MB, about 33 µs at 2 TB/s; at this size launch overhead (a few µs) is already visible.

## Pitfalls

- **Strict inequality**: a pixel equal to $\theta$ maps to 0. Since the
  output must match exactly, `>=` fails on integer-valued inputs.
- **Argument order**: `(input_image, threshold_value, output_image, height, width)`.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Grayscale](../grayscale/), [Edge Detect](../edge-detect/), [Histogram](../histogram/).

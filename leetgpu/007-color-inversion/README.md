---
title: Color Inversion
platform: LeetGPU
upstream: easy/7_color_inversion
url: https://leetgpu.com/challenges/color-inversion
difficulty: easy
tags: [elementwise, vectorized, uint8, image]
status: solved
---

# Color Inversion

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/color-inversion)

## Problem

Invert the colours of an RGBA image **in place**. `image` holds
$H \times W$ pixels of 4 unsigned bytes (R, G, B, A) in row-major order
($1 \le W, H \le 4096$, $WH \le 8\,388\,608$; benchmark $H = 5120$,
$W = 4096$). Each of R, G, B is replaced by $255 - v$, and alpha is left
unchanged. The problem is trivial arithmetically. Its lesson is **access
granularity**: moving bytes one at a time wastes most of the memory system.

## Formulation

$$
\text{img}'_{p,c} =
\begin{cases}
255 - \text{img}_{p,c}, & c \in \{0, 1, 2\} \ (\text{R, G, B}) \\
\text{img}_{p,c}, & c = 3 \ (\text{A})
\end{cases}
\qquad \text{offset}(p, c) = 4p + c, \quad p = yW + x
$$

| Symbol | Meaning |
|---|---|
| $W,\ H$ | image width and height in pixels |
| $x,\ y$ | pixel column and row |
| $p$ | linear pixel index, $0 \le p < WH$ |
| $c$ | channel index: 0 = R, 1 = G, 2 = B, 3 = A |
| $\text{img}_{p,c}$ | byte value before inversion, $0..255$ |
| $\text{img}'_{p,c}$ | byte value after inversion |

For 8-bit values, $255 - v$ equals the bitwise complement $\lnot v$, so the
operation can also be written as `v ^ 0xFF`.

## Approach

One thread per **pixel**. The buffer is reinterpreted as `uchar4*`, so a
thread performs a single 4-byte load and a single 4-byte store:

1. `p = blockIdx.x * 256 + threadIdx.x`, guarded by `p < W*H`.
2. `uchar4 v = pixels[p]`, then `v.x = 255 - v.x` (and likewise `.y`, `.z`);
   `.w` (alpha) is untouched.
3. `pixels[p] = v`.

A warp then accesses $32 \times 4 = 128$ contiguous bytes, one fully used
transaction. With one thread per *byte* instead, each thread would issue a
1-byte access: 4× the instructions and load/store operations for the same
data. There would also be a branch per byte to skip alpha.

A further 4× could come from `uint4` (16 bytes = 4 pixels per thread) with
the mask trick `v ^ 0x00FFFFFF` per 32-bit word. At this size the one-pixel
version is already bandwidth-bound, so it is kept for clarity.

## Cost Analysis

$$
Q = 2 \cdot 4\,WH \ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM traffic: every pixel is read and written once (4 bytes each way) |
| $\beta$ | DRAM bandwidth |
| $T_{\min}$ | bandwidth lower bound |

At the benchmark size, $WH = 2.1\times10^7$ pixels and $Q = 168$ MB, so
$T_{\min} \approx 84\ \mu s$ at 2 TB/s. There is no meaningful arithmetic.

## Pitfalls

- **Modifying alpha.** The reference keeps channel 3 unchanged.
- **Alignment.** `uchar4` loads need 4-byte alignment. That holds for
  `cudaMalloc` buffers, which are 256-byte aligned.
- **In-place update.** Each thread reads and writes only its own pixel, so
  there are no read-after-write hazards between threads.

## Verification

Exact equality (integer data) on all LeetGPU cases in
[cuemu](../../tools/cuemu/README.md), including 1 × 1 images.

## Related

- [RGB to Grayscale](../066-rgb-to-grayscale/), [Vector Addition](../001-vector-add/).
- Tensara [Grayscale](../../tensara/grayscale/).

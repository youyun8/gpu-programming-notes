---
title: Image Histogram
platform: Tensara
upstream: histogram
url: https://tensara.org/problems/histogram
difficulty: easy
tags: [histogram, atomics, shared-memory, privatization]
status: solved
---

# Image Histogram

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/histogram)

## Problem

Count how many pixels of an $h\times w$ grayscale image fall in each of
$n_b$ bins ($n_b$ = 64, 128 or 256; images up to $4096\times4096$). Pixel
values are integers stored as floats; the reference clamps them to
$[0, n_b - 1]$ and uses `torch.bincount`. The counts are returned as
floats and compared exactly.

## Formulation

$$
\text{bin}(v) = \bigl\lfloor \min\bigl(\max(v, 0),\ n_b - 1\bigr) \bigr\rfloor, \qquad
H[k] = \sum_{i=0}^{h-1}\sum_{j=0}^{w-1} \mathbf{1}\bigl[\text{bin}(I_{ij}) = k\bigr]
$$

| Symbol | Meaning |
|---|---|
| $I_{ij}$ | pixel value |
| $n_b$ | number of bins (`num_bins`) |
| $\text{bin}(v)$ | bin index of value $v$ after clamping |
| $\mathbf{1}[\cdot]$ | indicator: 1 if the condition holds, else 0 |
| $H[k]$ | count of bin $k$, $0 \le k < n_b$ |

Privatization splits the count by block and sums the private copies:

$$
H[k] = \sum_{b=0}^{G-1} H_b[k], \qquad H_b[k] = \sum_{p \in \mathcal{P}_b} \mathbf{1}\bigl[\text{bin}(I_p) = k\bigr]
$$

| Symbol | Meaning |
|---|---|
| $G$ | number of blocks |
| $\mathcal{P}_b$ | pixels visited by block $b$ |
| $H_b$ | block $b$'s private histogram in shared memory |

## Approach

1. **`zeroBins`** clears the output (it is accumulated into).
2. **`histogramKernel`**: each block zeroes a shared `unsigned` histogram,
   then walks its pixels with a grid-stride loop, doing
   `atomicAdd(&s_hist[bin], 1)` in shared memory. Shared atomics are
   resolved in the SM and only contend within the block.
3. After a barrier, each block adds its **non-zero** bins to the global
   histogram with one float `atomicAdd` per bin.
4. A fallback does global atomics directly if $n_b > 8192$ (never used by
   the tests).

Float counts are exact up to $2^{24} = 16.7$ M, exactly $4096^2$, so the
largest test is right at the limit and still exact.

## Cost analysis

$$
Q = 4hw + 4n_b\ \text{bytes}, \qquad \#\text{global atomics} \le G\,n_b, \qquad \#\text{shared atomics} = hw
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes: read the image once |
| $G n_b$ | block-to-global merges (at most one per bin per block) |
| $hw$ | one shared-memory atomic per pixel |

At $4096^2$: 67 MB, ~34 µs at 2 TB/s. With 64 bins and natural images,
many lanes of a warp hit the same bin and shared atomics serialize; one
private histogram per warp (bins × warps in shared memory) would cut that
contention further.

## Pitfalls

- **Zero the output**: global atomics accumulate onto whatever is there.
- **Clamp before casting**: values outside $[0, n_b - 1]$ must land in the
  end bins, as `torch.clamp` does.
- **Exactness**: the checker compares exactly, so counts must be integers
  (no averaging, no float drift beyond $2^{24}$).

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Grayscale](../grayscale/), [Threshold](../threshold/),
  LeetGPU [Histogramming](../../leetgpu/013-histogramming/).

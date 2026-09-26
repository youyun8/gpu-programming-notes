---
title: 1D Convolution
platform: Tensara
upstream: conv-1d
url: https://tensara.org/problems/conv-1d
difficulty: easy
tags: [convolution, shared-memory, tiling]
status: solved
---

# 1D Convolution

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/conv-1d)

## Problem

"Same" 1-D convolution of a float32 signal of length $N$ with an odd
kernel of $K$ taps and zero padding $r = (K-1)/2$ on both sides. As in
PyTorch's `conv1d`, this is a cross-correlation: the kernel is **not**
flipped. The tests use huge kernels ($K = 8191$) on $N$ = 32 K … 512 K, so
the work is $N K$ multiply-adds, not a memory stream. The check is
`rtol = 2e-4`, `atol = 5e-3`.

## Formulation

$$
C[i] = \sum_{j=0}^{K-1} \tilde{A}[\,i + j - r\,]\; B[j], \qquad
\tilde{A}[t] = \begin{cases} A[t], & 0 \le t < N \\ 0, & \text{otherwise} \end{cases}, \qquad
r = \frac{K-1}{2}
$$

| Symbol | Meaning |
|---|---|
| $A$ | Input signal of length $N$ |
| $\tilde{A}$ | $A$ extended with zeros (the padding) |
| $B$ | Kernel (filter) of $K$ taps, $K$ odd |
| $r$ | Kernel radius; the kernel is centred on $i$ |
| $C$ | Output of length $N$ |
| $i$ | Output index; $j$ tap index |

Output $i$ touches the input window $[\,i - r,\ i + r\,]$. A block that
produces the outputs $[b, b + T)$ therefore needs the window

$$
\bigl[\,b - r,\ b + T - 1 + r\,\bigr], \qquad \text{length } T + K - 1
$$

| Symbol | Meaning |
|---|---|
| $b$ | First output of the block |
| $T$ | Outputs per block (1024 here) |

## Approach

1. **Tile the outputs**: a block of 256 threads owns $T = 1024$ outputs,
   4 per thread, strided by 256 so that at any instant the 32 lanes of a
   warp read 32 consecutive shared words (no bank conflicts).
2. **Chunk the taps**: with $K = 8191$ the full window ($1024 + 8190$
   floats) would not fit comfortably in shared memory. The taps are
   processed in chunks of 2048. Per chunk the block loads the 2048 taps
   and the matching $1024 + 2047$ inputs (zero-filled outside $[0, N)$),
   synchronizes, and runs the FMA loop out of shared memory. Shared memory
   is 8 KB + 12 KB, independent of $K$.
3. **Inner loop**: $B[j]$ is read once per $j$ as a broadcast, then four
   `fmaf`s with four shared loads. Each input element is read from DRAM
   about $\lceil K/2048 \rceil$ times in total, which is negligible next to
   the $NK$ FMAs.

## Cost Analysis

$$
W = 2NK\ \text{flops}, \qquad Q \approx 4N\left(2 + \left\lceil \tfrac{K}{2048} \right\rceil\right) + 4K\left\lceil \tfrac{N}{T} \right\rceil\ \text{bytes}, \qquad
I = \frac{W}{Q}
$$

| Symbol | Meaning |
|---|---|
| $W$ | floating-point operations (one FMA = 2 flops) |
| $Q$ | DRAM (or L2) bytes: input windows, output, kernel re-reads per block |
| $I$ | Arithmetic intensity, thousands of flops per byte here |

At $N = 524288$, $K = 8191$: $W = 8.6$ GFLOP. The kernel is compute-bound,
but on the **shared-memory load** port rather than the FMA units: every FMA
needs one `LDS`, and an SM issues fewer shared loads than FMAs per cycle.
The next step is register blocking: give each thread several *adjacent*
outputs and slide a register window, so that one shared load feeds several
FMAs.

## Pitfalls

- **Cross-correlation, not convolution**: $B[j]$ multiplies
  $A[i + j - r]$; flipping the kernel is wrong.
- **Accumulation error**: 8191 fp32 terms per output; the tolerance
  (`atol = 5e-3`) allows for the different summation order of cuDNN.
- **Bounds**: $i + j - r$ is negative near the start; use signed 64-bit
  arithmetic for global indices.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Conv 2D](../conv-2d/), [Conv Square 3D](../conv-square-3d/),
  LeetGPU [1D Convolution](../../leetgpu/009-1d-convolution/).

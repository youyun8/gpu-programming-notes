---
title: 2D FFT
platform: LeetGPU
upstream: medium/78_2d_fft
url: https://leetgpu.com/challenges/2d-fft
difficulty: medium
tags: [fft, transpose, shared-memory, complex]
status: solved
---

# 2D FFT

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/2d-fft)

## Problem

The 2-D DFT of an $M\times N$ complex float32 signal, stored as interleaved
(re, im) pairs in row-major order ($M, N \le 4096$; benchmark
$M = N = 2048$; tolerance `1e-2`). It matches `torch.fft.fft2`. The 2-D DFT
is separable: 1-D DFTs along rows, then along columns. The engineering
question is how to make both passes read memory contiguously.

## Formulation

$$
X_{uv} = \sum_{m=0}^{M-1}\sum_{n=0}^{N-1} x_{mn}\,\omega_M^{um}\,\omega_N^{vn}
= \sum_{m=0}^{M-1} \omega_M^{um}\underbrace{\sum_{n=0}^{N-1} x_{mn}\,\omega_N^{vn}}_{Y_{mv}\ (\text{row DFTs})}, \qquad \omega_L = e^{-2\pi i/L}
$$

| Symbol | Meaning |
|---|---|
| $M,\ N$ | Rows and columns |
| $x_{mn}$ | Input sample, complex; real part at `2(mN+n)`, imaginary at `2(mN+n)+1` |
| $X_{uv}$ | Spectrum coefficient |
| $\omega_L$ | Primitive $L$-th root of unity |
| $Y_{mv}$ | Intermediate: the 1-D DFT of row $m$ |

**Row–column algorithm:** DFT each row ($M$ transforms of length $N$), then
DFT each column of the result ($N$ transforms of length $M$). The cost is
$O(MN\log(MN))$ with FFTs.

### Radix-2 Decimation in Time (in Shared Memory)

For power-of-two $L$: permute the input into **bit-reversed** order, then
run $\log_2 L$ butterfly stages. In the stage with half-size $h$:

$$
\begin{aligned}
u &= s[i_0], \quad v = s[i_1]\cdot\omega_{L}^{\,p\cdot L/(2h)}, \qquad i_0 = 2hg + p,\ \ i_1 = i_0 + h \\
s[i_0] &\leftarrow u + v, \qquad s[i_1] \leftarrow u - v
\end{aligned}
$$

| Symbol | Meaning |
|---|---|
| $s$ | The row held in shared memory |
| $h$ | Half the current sub-transform size: $1, 2, 4, \dots, L/2$ |
| $g,\ p$ | Butterfly group and position within the group ($0 \le p < h$) |
| $\omega_L^{p L/(2h)}$ | Twiddle factor $= e^{-2\pi i p/(2h)}$ |

## Approach

1. **`fftRows`** over the $M$ rows (one block per row, 512 threads):
   - *power of two*: bit-reversed load into shared memory
     (`__brev(i) >> (32 - log2 L)`), then $\log_2 L$ stages of $L/2$
     butterflies, with a barrier between stages;
   - *otherwise*: a direct $O(L^2)$ DFT per output (only small non-power-of-two
     sizes appear in the tests).
2. **Transpose** ($M\times N \to N\times M$) with the tiled 32 × 32
   shared-memory transpose (padding against bank conflicts), so the columns
   become contiguous rows.
3. **`fftRows`** over the $N$ rows of the transposed matrix (length $M$).
4. **Transpose back** into `spectrum`.

Twiddles are computed with `sincospif(-2(k mod L)/L)`. The integer reduction
keeps the argument in $[0, 2)$, and `sincospif` avoids multiplying by a
rounded $\pi$. Both matter at $L = 4096$.

A row of 4096 complex values is 32 KB of shared memory. The launcher
opts in to the dynamic size explicitly.

## Cost Analysis

$$
W \approx 5MN\log_2(MN), \qquad Q \approx \underbrace{2\cdot 8MN}_{\text{row FFTs}}\cdot 2 + \underbrace{2\cdot 8MN}_{\text{transposes}}\cdot 2 = 64MN\ \text{bytes}
$$

| Symbol | Meaning |
|---|---|
| $W$ | Real FLOPs (radix-2 estimate) |
| $Q$ | DRAM bytes: each FFT pass and each transpose reads and writes the whole $M\times N$ complex array (8 bytes per element) |

$2048^2$: $W \approx 0.46$ GFLOP and $Q \approx 270$ MB, i.e. ≈ 135 µs of traffic.
Each row FFT happens entirely in shared memory, so global traffic is only 2
per pass. Fusing the column FFT with the transposes (processing column
panels directly through shared memory) would save half of the traffic.

## Pitfalls

- **Strided column FFTs.** Reading columns directly means accesses $N$
  elements apart, which is uncoalesced. The transposes fix that.
- **Twiddle precision** at large $L$ (see above).
- **Non-power-of-two sizes** fall back to $O(L^2)$. For large arbitrary sizes,
  use Bluestein (see [FFT](../039-fast-fourier-transform/)).

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-2`,
including $1 \times N$, $M \times 1$ and non-power-of-two shapes.

## Related

- [Fast Fourier Transform (1-D, any length)](../039-fast-fourier-transform/), [Matrix Transpose](../003-matrix-transpose/).

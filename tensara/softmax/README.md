---
title: Softmax
platform: Tensara
upstream: softmax
url: https://tensara.org/problems/softmax
difficulty: medium
tags: [softmax, online-softmax, strided-reduction]
status: solved
---

# Softmax

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/softmax)

## Problem

Softmax along an arbitrary dimension `dim` of a float32 tensor of any
rank, with the shape given as an array of `ndim` sizes. The tests mix
contiguous reductions ($(4, 256, 256, 256)$ along dim 3) with strided ones
($(8, 1024, 1024)$ along dim 1, $(256, 50, 50)$ along dim 0). The check
is `rtol = 2e-3`, `atol = 1e-4`.

## Formulation

View the tensor as three axes, as in [Argmax](../argmax/):

$$
O = \prod_{k<d} S_k, \qquad R = S_d, \qquad I = \prod_{k>d} S_k, \qquad
x[o, j, i] = \text{in}\bigl[(oR + j)I + i\bigr]
$$

$$
\text{out}[o, j, i] = \frac{e^{x[o,j,i] - m_{oi}}}{s_{oi}}, \qquad
m_{oi} = \max_{j} x[o, j, i], \qquad s_{oi} = \sum_{j=0}^{R-1} e^{x[o,j,i] - m_{oi}}
$$

| Symbol | Meaning |
|---|---|
| $S_k$ | Size of axis $k$; $d$ is `dim` |
| $O, R, I$ | Outer size, reduced length, inner size (stride of the reduced axis) |
| $x[o, j, i]$ | Element at outer $o$, reduced index $j$, inner $i$ |
| $m_{oi}$ | Maximum along the reduced axis |
| $s_{oi}$ | Normaliser: sum of shifted exponentials |

Both $m$ and $s$ come from one pass with the online merge
$(m_1, s_1)\oplus(m_2, s_2) = \bigl(M, s_1e^{m_1-M} + s_2e^{m_2-M}\bigr)$,
$M = \max(m_1, m_2)$ (see [Log Softmax](../log-softmax/)).

| Symbol | Meaning |
|---|---|
| $\oplus$ | Associative merge of partial (max, sum) pairs |

## Approach

The shape array may live on the host or the device, so it is copied with
`cudaMemcpyDefault`; $O$, $R$, $I$ are computed on the host.

- **$I = 1$** (contiguous rows): `softmaxRows`, one warp per row. Lanes
  merge strided elements, a shuffle butterfly combines the 32 pairs, then
  the lanes write $e^{x - m}/s$. Two reads and one write per element.
- **$I > 1$** (strided): `softmaxStrided`, one thread per $(o, i)$
  column. The thread walks $j$ with stride $I$; neighbouring threads have
  neighbouring $i$, so each step is a coalesced warp access. A second walk
  writes the outputs.

## Cost Analysis

$$
Q = 12\,ORI\ \text{bytes (two reads, one write)}, \qquad \#\exp \approx 2\,ORI
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM (or L2) bytes |
| #exp | Exponentials: one in the online pass, one in the write pass |

Largest case $64\times128^3$: 537 MB of input, ~0.8 ms of traffic at
2 TB/s. For small $R$ (e.g. $(128, 10)$ along dim 1), a warp per row wastes
22 of 32 lanes; several rows per warp would help, but these cases are
tiny.

## Pitfalls

- **Strided case with few columns**: $(256, 50, 50)$ along dim 0 has only
  2500 columns, i.e. 2500 threads, so the GPU is mostly idle; the
  reduction could be split along $j$.
- **`shape` pointer kind**: use `cudaMemcpyDefault`, never dereference it
  on the host.
- **Stability**: subtract the maximum.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Log Softmax](../log-softmax/), [Argmax](../argmax/),
  [Scaled Dot-Product Attention](../scaled-dot-attention/),
  LeetGPU [Softmax](../../leetgpu/005-softmax/).

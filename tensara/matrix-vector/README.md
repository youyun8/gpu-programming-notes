---
title: Matrix Vector Multiplication
platform: Tensara
upstream: matrix-vector
url: https://tensara.org/problems/matrix-vector
difficulty: easy
tags: [gemv, warp-per-row, float4, bandwidth-bound]
status: solved
---

# Matrix Vector Multiplication

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/matrix-vector)

## Problem

Matrix–vector product $\mathbf{c} = A\mathbf{b}$ for $A$ of size $M\times K$
($M$ = 4096 … 9216, $K = 4096$). The check is `rtol = 2e-4`, `atol = 3e-3`.
Unlike GEMM, every element of $A$ is used exactly once, so the problem is
about streaming $A$ at full bandwidth.

## Formulation

$$
c_i = \sum_{k=0}^{K-1} A_{ik}\,b_k, \qquad 0 \le i < M
$$

| Symbol | Meaning |
|---|---|
| $A$ | Matrix, $M\times K$, row-major |
| $\mathbf{b}$ | Input vector, length $K$ |
| $\mathbf{c}$ | Output vector, length $M$ |

Per row, lane $\ell$ of a warp computes a strided partial sum, then the
warp reduces:

$$
p_\ell = \sum_{q\,:\,q \equiv \ell \pmod{32}} \mathbf{a}^{(4)}_{iq}\cdot\mathbf{b}^{(4)}_q, \qquad c_i = \sum_{\ell=0}^{31} p_\ell
$$

| Symbol | Meaning |
|---|---|
| $\ell$ | Lane index |
| $\mathbf{a}^{(4)}_{iq}, \mathbf{b}^{(4)}_q$ | The $q$-th `float4` of row $i$ and of $\mathbf{b}$ |
| $p_\ell$ | Lane partial sum |

## Approach

1. **One warp per row**, 8 warps per block. Each lane reads `float4`s at
   stride 32, so one warp instruction reads 512 contiguous bytes of $A$.
2. **$\mathbf{b}$ stays in cache**: 16 KB, read by every warp, served by
   L1/L2 after the first touch.
3. **Shuffle reduction** (`__shfl_down_sync`, 5 steps) and lane 0 writes
   $c_i$.
4. A scalar fallback handles $K \bmod 4 \ne 0$, where rows are not 16-byte
   aligned.

## Cost Analysis

$$
Q = 4MK + 4K + 4M\ \text{bytes}, \qquad W = 2MK, \qquad I = \frac{W}{Q} \approx \frac{1}{2}\ \tfrac{\text{flop}}{\text{byte}}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes, dominated by reading $A$ once |
| $W$ | Flops |
| $I$ | Arithmetic intensity: far below the ridge point, so bandwidth-bound |

At $9216\times4096$: 151 MB, about 75 µs at 2 TB/s.

## Pitfalls

- **Do not use the SGEMM** with $N = 1$: its tiles would be 98 % padding.
- **Alignment**: the `float4` path requires $K \bmod 4 = 0$ (and 16-byte
  aligned buffers, which `cudaMalloc` guarantees).

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [NVFP4 GEMV](../nvfp4-gemv/), [Matrix Multiplication](../matrix-multiplication/),
  LeetGPU [Dot Product](../../leetgpu/017-dot-product/).

---
title: MXFP8 GEMM
platform: Tensara
upstream: mxfp8-gemm
url: https://tensara.org/problems/mxfp8-gemm
difficulty: hard
tags: [matmul, mxfp8, block-scaled, low-precision]
status: solved
---

# MXFP8 GEMM

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/mxfp8-gemm)

## Problem

Compute $C = \hat{A}\hat{B}^{\mathsf T}$ in FP32 where $A$ ($M\times K$) and
$B$ ($N\times K$) are MXFP8 tensors: E4M3 elements with E8M0 scales per 32
elements, the scales in the **swizzled 128×4 layout** produced by
`to_mx(..., is_swizzled_scales=True)`. The reference is
`torch._scaled_mm`. The check is `rtol = 2e-2`, `atol = 5e-2`.

## Formulation

$$
c_{ij} = \sum_{\ell=0}^{K-1} \hat{A}_{i\ell}\,\hat{B}_{j\ell}, \qquad
\hat{A}_{i\ell} = \operatorname{e4m3}(q^A_{i\ell})\,2^{u^A_{i,\lfloor\ell/32\rfloor} - 127}, \qquad \hat{B}_{j\ell} = \operatorname{e4m3}(q^B_{j\ell})\,2^{u^B_{j,\lfloor\ell/32\rfloor} - 127}
$$

| Symbol | Meaning |
|---|---|
| $\hat{A}$ | dequantized $A$, $M\times K$ |
| $\hat{B}$ | dequantized $B$, stored $N\times K$ (so the product is $\hat{A}\hat{B}^{\mathsf T}$, an "NT" GEMM) |
| $c$ | output, $M\times N$ (FP32) |
| $u^A, u^B$ | E8M0 scale bytes (swizzled) |

Because every block of 32 elements shares one scale, the sum can be
regrouped by blocks, which is what tensor-core block-scaled MMAs do:

$$
c_{ij} = \sum_{\beta=0}^{K/32 - 1} \sigma^{A}_{i\beta}\,\sigma^{B}_{j\beta} \sum_{\ell \in \beta} x^{A}_{i\ell}\,x^{B}_{j\ell}
$$

| Symbol | Meaning |
|---|---|
| $\beta$ | block index along $K$ |
| $\sigma^A_{i\beta}, \sigma^B_{j\beta}$ | the two block scales |
| $x^A, x^B$ | the decoded element values (before scaling) |

**E4M3 (FP8)** has 1 sign, 4 exponent and 3 mantissa bits, bias 7,
no infinities, and codes `0x7F`/`0xFF` are NaN:

$$
\operatorname{e4m3}(b) = (-1)^{s}\cdot\begin{cases} \dfrac{f}{8}\cdot 2^{-6}, & e = 0 \ (\text{subnormal}) \\ \Bigl(1 + \dfrac{f}{8}\Bigr) 2^{e - 7}, & 1 \le e \le 15 \end{cases}, \qquad \lvert\operatorname{e4m3}\rvert \le 448
$$

| Symbol | Meaning |
|---|---|
| $b$ | the byte |
| $s, e, f$ | sign bit, 4-bit exponent field, 3-bit mantissa field |
| 448 | largest finite value ($e = 15$, $f = 6$) |

**E8M0** (the MX block scale) is a bare power of two:

$$
\operatorname{e8m0}(u) = 2^{\,u - 127}, \qquad u \in [0, 254], \quad u = 255 \Rightarrow \text{NaN}
$$

| Symbol | Meaning |
|---|---|
| $u$ | the scale byte (a biased exponent) |

**Swizzled scale layout.** Block-scaled tensor-core MMAs (cuBLAS /
CUTLASS, TorchAO `is_swizzled_scales=True`, FlashInfer) store the scale
matrix of $R$ rows and $C$ scale columns in $128\times4$ atoms of 512 bytes:

$$
\operatorname{idx}(r, c) = \Bigl(\bigl\lfloor \tfrac{r}{128} \bigr\rfloor \Bigl\lceil \tfrac{C}{4} \Bigr\rceil + \bigl\lfloor \tfrac{c}{4} \bigr\rfloor\Bigr)\cdot 512 +
(r \bmod 32)\cdot 16 + \Bigl\lfloor \tfrac{r \bmod 128}{32} \Bigr\rfloor\cdot 4 + (c \bmod 4)
$$

| Symbol | Meaning |
|---|---|
| $r$ | matrix row |
| $c$ | scale column (block index along $K$) |
| $C$ | number of scale columns, $K/\text{block}$ |
| idx | byte offset of scale $(r, c)$ (`swizzledScaleIndex`) |

Inside an atom, rows $r, r+32, r+64, r+96$ are interleaved so that one
16-byte load gives a thread the 4 scales of 4 rows it needs.

## Approach

A block-scaled version of the register-blocked SGEMM used across the
Tensara matmul pages (`blockScaledGemm`):

1. **$64\times64$ output tile**, 256 threads, $4\times4$ outputs per thread.
2. **K-slices of 32 (one scale block)**: while staging the $A$ and $B$ panels into shared
   memory, each thread decodes the element code (`e4m3ToFloat`) and multiplies by
   its block scale, looked up through the swizzled index. The tiles hold
   FP32, so the inner loop is the plain FMA outer product.
3. **Epilogue**: plain FP32 store (`global_scale = 1`).

The dequantized matrices are never written to global memory; the only
extra cost compared with an FP32 GEMM is the decode work while staging.
On Blackwell (sm_100) the same data would feed `tcgen05.mma` block-scaled
instructions directly, which read these swizzled scale layouts in
hardware; this portable kernel uses CUDA cores instead.

## Cost analysis

$$
W = 2MNK, \qquad Q_{\min} = 1\,(MK + NK) + \frac{MK + NK}{32} + 4\,MN\ \text{bytes}
$$

| Symbol | Meaning |
|---|---|
| $W$ | flops |
| $Q_{\min}$ | compulsory DRAM bytes: packed operands, scales, output |

FP8 operands are 4× smaller than FP32, so for moderate sizes the kernel is
even more compute-bound than an SGEMM; the decode (a few integer ops per
element staged) is amortised over 64 FMAs per staged value.

## Pitfalls

- **Swizzled scales**: indexing them row-major reads the wrong scale for
  every block with $r \bmod 128 \ge 32$.
- **$B$ is $N\times K$**: an "NT" product.
- **Tolerance**: the reference accumulates on tensor cores in a different
  order; `atol = 5e-2` absorbs it.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [MXFP4 GEMM](../mxfp4-gemm/), [NVFP4 GEMM](../nvfp4-gemm/), [MXFP8 Quantize](../mxfp8-quantize/),
  [Matrix Multiplication](../matrix-multiplication/).

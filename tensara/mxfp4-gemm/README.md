---
title: MXFP4 GEMM
platform: Tensara
upstream: mxfp4-gemm
url: https://tensara.org/problems/mxfp4-gemm
difficulty: hard
tags: [matmul, mxfp4, block-scaled, low-precision]
status: solved
---

# MXFP4 GEMM

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/mxfp4-gemm)

## Problem

Compute $C = \hat{A}\hat{B}^{\mathsf T}$ in FP32 where $A$ ($M\times K$) and
$B$ ($N\times K$) are MXFP4 tensors: packed E2M1 elements with E8M0 scales
per 32 elements in the swizzled 128×4 layout. The reference is
`torch._scaled_mm`. The check is `rtol = 2e-2`, `atol = 5e-2`.

## Formulation

$$
c_{ij} = \sum_{\ell=0}^{K-1} \hat{A}_{i\ell}\,\hat{B}_{j\ell}, \qquad
\hat{A}_{i\ell} = \operatorname{e2m1}(c^A_{i\ell})\,2^{u^A_{i,\lfloor\ell/32\rfloor} - 127}, \qquad \hat{B}_{j\ell} = \operatorname{e2m1}(c^B_{j\ell})\,2^{u^B_{j,\lfloor\ell/32\rfloor} - 127}
$$

| Symbol | Meaning |
|---|---|
| $\hat{A}$ | dequantized $A$, $M\times K$ |
| $\hat{B}$ | dequantized $B$, stored $N\times K$ (so the product is $\hat{A}\hat{B}^{\mathsf T}$, an "NT" GEMM) |
| $c$ | output, $M\times N$ (FP32) |
| $c^A, c^B$ | 4-bit codes, two per byte (low nibble = even $\ell$) |
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

**E2M1 (FP4)** has 1 sign, 2 exponent and 1 mantissa bit (bias 1). Its
eight magnitudes and the decode rule are

$$
\operatorname{e2m1}(c) = (-1)^{c_3}\cdot\begin{cases} \tfrac{1}{2}\,m, & m < 4 \\ (2 + (m \bmod 2))\cdot 2^{\lfloor m/2 \rfloor - 2}, & m \ge 4 \end{cases}
\in \pm\{0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6\}, \qquad m = c \mathbin{\&} 7
$$

| Symbol | Meaning |
|---|---|
| $c$ | 4-bit code; two codes per byte, element $2i$ in the **low** nibble |
| $c_3$ | sign bit (bit 3) |
| $m$ | 3-bit magnitude code, 0 … 7 |

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
   memory, each thread decodes the element code (the nibble, then `e2m1ToFloat`) and multiplies by
   its block scale, looked up through the swizzled index. The tiles hold
   FP32, so the inner loop is the plain FMA outer product.
3. **Epilogue**: plain FP32 store.

The dequantized matrices are never written to global memory; the only
extra cost compared with an FP32 GEMM is the decode work while staging.
On Blackwell (sm_100) the same data would feed `tcgen05.mma` block-scaled
instructions directly, which read these swizzled scale layouts in
hardware; this portable kernel uses CUDA cores instead.

## Cost analysis

$$
W = 2MNK, \qquad Q_{\min} = 0.5\,(MK + NK) + \frac{MK + NK}{32} + 4\,MN\ \text{bytes}
$$

| Symbol | Meaning |
|---|---|
| $W$ | flops |
| $Q_{\min}$ | compulsory DRAM bytes: packed operands, scales, output |

FP4 operands take 1/8 of the FP32 bytes, so operand traffic is small and
the FP32 output often dominates DRAM traffic for small $K$. Every E2M1
value times a power-of-two scale is exact in FP32, so the only rounding is
in the accumulation.

## Pitfalls

- **Nibble order** and **swizzled scales**, as in the other MX pages.
- **Byte addressing**: row $i$ of the packed payload starts at byte
  $iK/2$.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [MXFP8 GEMM](../mxfp8-gemm/), [NVFP4 GEMM](../nvfp4-gemm/), [MXFP4 Quantize](../mxfp4-quantize/),
  LeetGPU [INT4 MatMul](../../leetgpu/081-int4-matmul/).

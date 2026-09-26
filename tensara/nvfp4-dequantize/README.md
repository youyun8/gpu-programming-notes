---
title: NVFP4 Dequantization
platform: Tensara
upstream: nvfp4-dequantize
url: https://tensara.org/problems/nvfp4-dequantize
difficulty: medium
tags: [quantization, nvfp4, low-precision, elementwise]
status: solved
---

# NVFP4 Dequantization

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/nvfp4-dequantize)

## Problem

Expand an NVFP4 matrix (packed E2M1, swizzled E4M3 block scales, global
factor $g$) to FP32, with FlashInfer's `e2m1_and_ufp8sf_scale_to_float`
semantics. Sizes up to $8192\times4096$; `rtol = atol = 1e-3`.

## Formulation

**NVFP4** uses 16-element blocks along $K$ with a two-level scale: an
FP8 (E4M3) scale per block plus one FP32 global factor per tensor, which
moves the whole tensor into the range E4M3 × E2M1 can represent
($448\times6 = 2688$). The dequantized value of element $\ell$ of row $i$ is

$$
\hat{a}_{i\ell} = \frac{\operatorname{e2m1}(c_{i\ell})\cdot\operatorname{e4m3}\bigl(s_{i,\lfloor\ell/16\rfloor}\bigr)}{g}
$$

| Symbol | Meaning |
|---|---|
| $c_{i\ell}$ | 4-bit E2M1 code (two per byte, low nibble first) |
| $s_{i\beta}$ | E4M3 scale byte of block $\beta$, stored in the swizzled layout |
| $g$ | Global encode factor `sf_g` (FP32); $1/g$ is the global decode scale |
| $\hat{a}_{i\ell}$ | The value the encoding represents |

$$
\text{out}_{i\ell} = \operatorname{e2m1}(c_{i\ell})\cdot\operatorname{e4m3}\bigl(s[\operatorname{idx}(i, \lfloor\ell/16\rfloor)]\bigr)\cdot\frac{1}{g}
$$

| Symbol | Meaning |
|---|---|
| out | FP32 result, $M\times K$ |
| idx | Swizzled scale index (below) |

**E2M1 (FP4)** has 1 sign, 2 exponent and 1 mantissa bit (bias 1). Its
eight magnitudes and the decode rule are

$$
\operatorname{e2m1}(c) = (-1)^{c_3}\cdot\begin{cases} \tfrac{1}{2}\,m, & m < 4 \\ (2 + (m \bmod 2))\cdot 2^{\lfloor m/2 \rfloor - 2}, & m \ge 4 \end{cases}
\in \pm\{0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6\}, \qquad m = c \mathbin{\&} 7
$$

| Symbol | Meaning |
|---|---|
| $c$ | 4-bit code; two codes per byte, element $2i$ in the **low** nibble |
| $c_3$ | Sign bit (bit 3) |
| $m$ | 3-bit magnitude code, 0 … 7 |

**E4M3 (FP8)** has 1 sign, 4 exponent and 3 mantissa bits, bias 7,
no infinities, and codes `0x7F`/`0xFF` are NaN:

$$
\operatorname{e4m3}(b) = (-1)^{s}\cdot\begin{cases} \dfrac{f}{8}\cdot 2^{-6}, & e = 0 \ (\text{subnormal}) \\ \Bigl(1 + \dfrac{f}{8}\Bigr) 2^{e - 7}, & 1 \le e \le 15 \end{cases}, \qquad \lvert\operatorname{e4m3}\rvert \le 448
$$

| Symbol | Meaning |
|---|---|
| $b$ | The byte |
| $s, e, f$ | Sign bit, 4-bit exponent field, 3-bit mantissa field |
| 448 | Largest finite value ($e = 15$, $f = 6$) |

**Swizzled scale layout.** Block-scaled tensor-core MMAs (cuBLAS /
CUTLASS, TorchAO `is_swizzled_scales=True`, FlashInfer) store the scale
matrix of $R$ rows and $C$ scale columns in $128\times4$ atoms of 512 bytes:

$$
\operatorname{idx}(r, c) = \Bigl(\bigl\lfloor \tfrac{r}{128} \bigr\rfloor \Bigl\lceil \tfrac{C}{4} \Bigr\rceil + \bigl\lfloor \tfrac{c}{4} \bigr\rfloor\Bigr)\cdot 512 +
(r \bmod 32)\cdot 16 + \Bigl\lfloor \tfrac{r \bmod 128}{32} \Bigr\rfloor\cdot 4 + (c \bmod 4)
$$

| Symbol | Meaning |
|---|---|
| $r$ | Matrix row |
| $c$ | Scale column (block index along $K$) |
| $C$ | Number of scale columns, $K/\text{block}$ |
| idx | Byte offset of scale $(r, c)$ (`swizzledScaleIndex`) |

Inside an atom, rows $r, r+32, r+64, r+96$ are interleaved so that one
16-byte load gives a thread the 4 scales of 4 rows it needs.

## Approach

One thread per byte: decode the two nibbles, fetch the block's E4M3
scale through the swizzled index (8 threads share it, a cache hit),
multiply by $\operatorname{e4m3}(s)$ and by $1/g$ (computed once on the
host), and store a `float2`.

## Cost Analysis

$$
Q = 0.5\,MK + \frac{MK}{16}\ (\text{read}) + 4MK\ (\text{write})\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes, dominated by the FP32 output |
| $\beta$ | DRAM bandwidth |

About $4.56\,MK$ bytes, dominated by the FP32 output.

## Pitfalls

- **Scales are already swizzled**: do not swizzle again, do not read them
  row-major.
- **Global factor is an *encode* factor**: divide by $g$.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [NVFP4 Quantize](../nvfp4-quantize/), [MXFP4 Dequantize](../mxfp4-dequantize/).

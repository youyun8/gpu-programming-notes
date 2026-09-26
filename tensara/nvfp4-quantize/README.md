---
title: NVFP4 Quantization
platform: Tensara
upstream: nvfp4-quantize
url: https://tensara.org/problems/nvfp4-quantize
difficulty: medium
tags: [quantization, nvfp4, low-precision, half-warp]
status: solved
---

# NVFP4 Quantization

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/nvfp4-quantize)

## Problem

Quantize an $M\times K$ **FP16** matrix to NVFP4 given the global
encode factor $g$ (`sf_g`): packed E2M1 elements, one E4M3 scale per 16
elements in the swizzled 128×4 layout, following FlashInfer's
`nvfp4_quantize`. The check (after dequantizing both sides) is
`rtol = atol = 1e-3`.

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

Quantization of block $\beta$ of row $i$:

$$
\alpha_{i\beta} = \max_{\ell\in\beta}\lvert a_{i\ell}\rvert, \qquad
s_{i\beta} = \operatorname{e4m3}_{\text{RNE,sat}}\Bigl(g\cdot\frac{\alpha_{i\beta}}{6}\Bigr), \qquad
c_{i\ell} = \operatorname{e2m1}_{\text{RNE,sat}}\Bigl(a_{i\ell}\cdot\frac{g}{\operatorname{e4m3}(s_{i\beta})}\Bigr)
$$

| Symbol | Meaning |
|---|---|
| $a_{i\ell}$ | FP16 input element, converted to FP32 |
| $\alpha_{i\beta}$ | Block absolute maximum |
| $\alpha/6$ | The decode scale that maps the block maximum to E2M1's largest value 6 |
| $\operatorname{e4m3}_{\text{RNE,sat}}$ | Round to nearest even, saturate to $\pm448$ ("satfinite") |
| $\operatorname{e2m1}_{\text{RNE,sat}}$ | Round to nearest even among the 8 magnitudes, saturate at 6 |

Using the *rounded* scale $\operatorname{e4m3}(s)$ in the element encode
(not $\alpha/6$) is what makes decode(encode(x)) consistent.

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

1. **`zeroScales`** clears the whole padded scale buffer (the swizzled
   layout pads rows to multiples of 128 and scale columns to multiples of
   4; padding must be 0).
2. **`quantizeNvfp4`**: a **half-warp (16 lanes) per block**. Each lane
   loads one FP16 element; a 4-step shuffle max over the 16 lanes gives
   $\alpha$; every lane computes the E4M3 scale byte (bit-exact RNE with
   saturation), decodes it back to FP32, encodes its own element, and the
   even lane packs itself with its odd neighbour into one byte. Lane 0 of
   the half-warp writes the scale at `swizzledScaleIndex(row, blk)`.

## Cost Analysis

$$
Q = 2MK\ (\text{read}) + \frac{MK}{2} + \frac{MK}{16}\ (\text{write})\ \text{bytes}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes: FP16 in, packed FP4 and FP8 scales out |

About $2.56\,MK$ bytes; bandwidth-bound.

## Pitfalls

- **FP16 input** (`float16*`), unlike the MX problems.
- **Satfinite E4M3** for the scale: $g\alpha/6$ can exceed 448.
- **Swizzled scales with zero padding**.
- **Divide by the decoded E4M3 scale**, not by $\alpha/6$.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [NVFP4 Dequantize](../nvfp4-dequantize/), [NVFP4 GEMM](../nvfp4-gemm/),
  [NVFP4 GEMV](../nvfp4-gemv/), [MXFP4 Quantize](../mxfp4-quantize/).

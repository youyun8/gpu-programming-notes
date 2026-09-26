---
title: MXFP8 Dequantization
platform: Tensara
upstream: mxfp8-dequantize
url: https://tensara.org/problems/mxfp8-dequantize
difficulty: easy
tags: [quantization, mxfp8, low-precision, elementwise]
status: solved
---

# MXFP8 Dequantization

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/mxfp8-dequantize)

## Problem

Expand an MXFP8 matrix (E4M3 bytes plus row-major E8M0 scales per 32
elements) to FP32 with TorchAO semantics. Sizes up to $8192\times4096$;
`rtol = atol = 1e-3`.

## Formulation

$$
\text{out}_{ij} = \operatorname{e4m3}\bigl(q_{ij}\bigr)\cdot\operatorname{e8m0}\bigl(u_{i,\lfloor j/32\rfloor}\bigr)
$$

| Symbol | Meaning |
|---|---|
| $q_{ij}$ | E4M3 byte of element $(i, j)$ |
| $u$ | scale bytes, $M\times K/32$ |
| out | FP32 result |

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

## Approach

One thread per element (grid-stride): decode the byte with integer
operations (`e4m3ToFloat`: exponent and mantissa fields, a separate path
for subnormals and NaN), multiply by the block scale, store. The E4M3
decode could also be a 256-entry lookup table in shared memory, but the
arithmetic version is already hidden behind the memory traffic.

## Cost analysis

$$
Q = 1\,MK + \frac{MK}{32}\ (\text{read}) + 4MK\ (\text{write})\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes, dominated by the FP32 output |
| $\beta$ | DRAM bandwidth |

At $8192\times4096$: ~169 MB, ~85 µs at 2 TB/s.

## Pitfalls

- **Subnormals**: $e = 0$ means $f/8\cdot2^{-6}$, not $(1 + f/8)2^{-7}$.
- **NaN** is `0x7F`/`0xFF` only; E4M3 has no infinities.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [MXFP8 Quantize](../mxfp8-quantize/), [MXFP8 GEMM](../mxfp8-gemm/), [MXFP4 Dequantize](../mxfp4-dequantize/).

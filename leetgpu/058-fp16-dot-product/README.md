---
title: FP16 Dot Product
platform: LeetGPU
upstream: medium/58_fp16_dot_product
url: https://leetgpu.com/challenges/fp16-dot-product
difficulty: medium
tags: [reduction, fp16, mixed-precision]
status: solved
---

# FP16 Dot Product

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/fp16-dot-product)

## Problem

Dot product of two fp16 vectors of length $N$, accumulated in fp32 and
returned as fp16 ($N \le 10^8$; benchmark $N = 10^8$; tolerance `0.05`).
Halving the bytes per element (compared with float32) halves the runtime of
a bandwidth-bound reduction, as long as the accumulation stays in fp32.

## Formulation

$$
s = \operatorname{fp16}\!\Bigl(\sum_{i=0}^{N-1}\operatorname{fp32}(a_i)\,\operatorname{fp32}(b_i)\Bigr)
$$

| Symbol | Meaning |
|---|---|
| $N$ | vector length |
| $a_i,\ b_i$ | fp16 inputs (10-bit mantissa, max 65 504) |
| $s$ | result, rounded once to fp16 |

**Why not accumulate in fp16?** fp16 has only 11 significant bits. Once the
running sum reaches about 2048, adding values below 1 has no effect at all
(swamping), and the sum stops growing. The reference therefore widens to
fp32 first, and a correct kernel must do the same.

## Approach

1. **`partialDots`** (≤ 1024 blocks): a grid-stride loop over `half2` pairs
   (one 32-bit load delivers two fp16 values). `__half22float2` widens both,
   and two `fmaf` accumulate in fp32. Thread 0 handles an odd trailing
   element. The block reduction runs in float64 into `g_partials`.
2. **`finalSum`**: one block sums the partials in float64 and converts to
   fp16 (`__float2half`, round to nearest).

## Cost analysis

$$
Q = 2 \cdot 2N = 4N\ \text{bytes}, \qquad T_{\min} = \frac{4N}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes (two fp16 vectors) |
| $\beta$ | DRAM bandwidth |

Benchmark: 400 MB, i.e. ≈ 200 µs at 2 TB/s. That is half the time of the
float32 [Dot Product](../017-dot-product/) at the same $N$.

## Pitfalls

- **`half2` alignment**: 4-byte alignment is required. Base pointers from
  `cudaMalloc` are fine.
- **Odd $N$**: exactly one thread must add the last element.
- **fp16 overflow of the result.** A sum above 65 504 becomes $\infty$, as in
  the reference.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `0.05`,
including odd $N$.

## Related

- [Dot Product](../017-dot-product/), [GEMM (fp16)](../022-gemm/).

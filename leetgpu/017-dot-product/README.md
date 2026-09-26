---
title: Dot Product
platform: LeetGPU
upstream: medium/17_dot_product
url: https://leetgpu.com/challenges/dot-product
difficulty: medium
tags: [reduction, two-pass, fma, deterministic]
status: solved
---

# Dot Product

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/dot-product)

## Problem

Dot product of two float32 vectors of length $N$ ($1 \le N \le 10^8$),
written to `result[0]` with tolerance `1e-5`. It is a reduction whose
per-element operation is a multiply-add instead of an add. The whole design
of [Reduction](../004-reduction/) carries over, with one extra input stream.

## Formulation

$$
s = \mathbf a \cdot \mathbf b = \sum_{i=0}^{N-1} a_i\, b_i
$$

| Symbol | Meaning |
|---|---|
| $N$ | vector length |
| $a_i,\ b_i$ | elements of the input vectors `A`, `B` (float32) |
| $s$ | scalar result, stored in `result[0]` |

Per thread, the products are accumulated with a **fused multiply-add**:

$$
\text{acc} \leftarrow \operatorname{fma}(a_i, b_i, \text{acc}) = \operatorname{round}(a_i b_i + \text{acc})
$$

| Symbol | Meaning |
|---|---|
| acc | the thread's running float32 partial sum |
| $\operatorname{fma}$ | fused multiply-add: one rounding for the product and the add together |
| round | rounding to the nearest float32 |

FMA is both faster (one instruction) and more accurate (one rounding
instead of two) than a separate multiply and add.

## Approach

Two kernels, as in [Reduction](../004-reduction/):

1. **`partialDots`.** Up to 1024 blocks × 256 threads. A grid-stride loop
   over `float4` pairs does 4 `fmaf` per iteration, and a scalar tail handles
   $N \bmod 4$. Each block reduces its threads in float64 (warp shuffles and
   one shared-memory hop) and writes one partial.
2. **`finalSum`.** One block adds the partials in float64 and rounds once.

The block and final levels are done in float64 for accuracy and
determinism. The benchmark uses $N = 5$, where launch overhead is the entire
cost. The same code also scales to $10^8$ elements at full bandwidth.

## Cost analysis

$$
Q = 8N \ \text{bytes}, \qquad W = 2N, \qquad I = \frac{2N}{8N} = \frac14 \ \text{FLOP/byte}, \qquad T_{\min} = \frac{8N}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes (read both vectors once) |
| $W$ | FLOPs (one multiply and one add per element) |
| $I$ | arithmetic intensity |
| $\beta$ | DRAM bandwidth |

This kernel is memory-bound for any GPU.

## Pitfalls

- **Single-kernel atomics.** `atomicAdd` into `result` works but makes the
  low bits nondeterministic.
- **At least one block.** For $N < 4$ the vectorised block count computes
  to 0, so it is clamped to 1 and the tail loop does all the work.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md), including
$N = 1..7$ (tail-only paths).

## Related

- [Reduction](../004-reduction/), [FP16 Dot Product](../058-fp16-dot-product/),
  [Sparse MV](../018-sparse-matrix-vector-multiplication/).
- Tensara [Cosine Similarity](../../tensara/cosine-similarity/).

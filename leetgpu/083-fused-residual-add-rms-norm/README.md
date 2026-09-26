---
title: Fused Residual Add and RMS Norm
platform: LeetGPU
upstream: medium/83_fused_residual_add_rms_norm
url: https://leetgpu.com/challenges/fused-residual-add-and-rms-norm
difficulty: medium
tags: [normalization, fusion, row-reduction, llm]
status: solved
---

# Fused Residual Add and RMS Norm

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/fused-residual-add-and-rms-norm)

## Problem

The "add & norm" step of LLaMA-style transformers, fused into one kernel:
add the sublayer output $x$ to the residual stream $r$, then RMS-normalise
each row and scale by a per-feature weight ($N, C \le 65\,536$,
$\varepsilon = 10^{-5}$; benchmark $N = C = 4096$). The intermediate $z = x + r$
must not be written to global memory. That is the point of the fusion.

## Formulation

$$
z_{ij} = x_{ij} + r_{ij}, \qquad
\operatorname{rms}_i = \sqrt{\frac1C\sum_{j=0}^{C-1} z_{ij}^2 + \varepsilon}, \qquad
y_{ij} = \frac{z_{ij}}{\operatorname{rms}_i}\, w_j
$$

| Symbol | Meaning |
|---|---|
| $N$ | number of tokens (rows) |
| $C$ | hidden dimension (columns) |
| $x_{ij}$ | sublayer output |
| $r_{ij}$ | residual stream |
| $z_{ij}$ | updated residual (kept only in registers or cache) |
| $\varepsilon$ | stability constant ($10^{-5}$) |
| $\operatorname{rms}_i$ | root mean square of row $i$ |
| $w_j$ | per-feature weight ($\gamma$) |
| $y_{ij}$ | normalised output |

## Approach

**One block of 256 threads per row:**

1. **Pass 1.** Stream the row, computing $z = x + r$ on the fly and
   accumulating $\sum z^2$ in registers. It uses `float4` loads when $C$ is a
   multiple of 4, and scalar loads otherwise. A block reduction (warp shuffles,
   then a shared-memory hop) produces the sum, and
   `inv_rms = rsqrtf(sum/C + eps)`.
2. **Pass 2.** Stream the row again, recompute $z$, and write
   $z\cdot\text{inv\_rms}\cdot w_j$. The row (at most 512 KB for two inputs)
   was just read, so the second read is served from L1/L2, not DRAM.

### What fusion saves

| Variant | DRAM traffic per element |
|---|---|
| Separate add kernel + RMSNorm kernel | read $x, r$, write $z$ (12 B) + read $z$ twice, write $y$ (12 B) = 24 B |
| Fused (this kernel) | read $x, r$ (8 B) + write $y$ (4 B) = 12 B |

That is half of the traffic, for an operation that runs twice per
transformer layer.

## Cost analysis

$$
Q = 12NC\ \text{bytes}, \qquad W \approx 5NC, \qquad T_{\min} = \frac{12NC}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes of the fused kernel |
| $W$ | FLOPs (add, square, accumulate, two multiplies) |
| $\beta$ | DRAM bandwidth |

Benchmark: 201 MB, i.e. ≈ 100 µs at 2 TB/s.

## Pitfalls

- **The mean is over $C$**, per row. It is not a global statistic, unlike
  [RMS Normalization](../050-rms-normalization/).
- **Rows longer than the cache.** For $C = 65\,536$ a row is 512 KB of
  inputs, which still fits in L2. For larger rows, pass 2 would re-read DRAM,
  and a single-pass variant that keeps $z$ in shared memory or registers
  would be needed.
- **Real LLM kernels** also *write back* $z$ as the new residual stream
  (in-place). The problem does not ask for it.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md), including
$C = 1$ and $C$ not divisible by 4 (the scalar path).

## Related

- [RMS Normalization](../050-rms-normalization/), [LLaMA Transformer Block](../093-llama-transformer-block/),
  [Layer Normalization](../113-layer-normalization/). Tensara [RMS Norm](../../tensara/rms-norm/).

---
title: RMS Normalization
platform: LeetGPU
upstream: medium/50_rms_normalization
url: https://leetgpu.com/challenges/rms-normalization
difficulty: medium
tags: [normalization, reduction, three-pass]
status: solved
---

# RMS Normalization

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/rms-normalization)

## Problem

RMS-normalise one float32 vector of length $N$ with a **scalar** scale
$\gamma$ and shift $\beta$ ($N \le 10^5$, $\varepsilon = 10^{-5}$; benchmark
$N = 10^5$; tolerance `1e-5`). Unlike LayerNorm, RMSNorm does not subtract
the mean. It is the normalisation used in LLaMA-style transformers (there
with a per-feature $\gamma$ and no $\beta$).

## Formulation

$$
\operatorname{rms} = \sqrt{\frac{1}{N}\sum_{i=0}^{N-1} x_i^2 + \varepsilon}, \qquad
y_i = \gamma\,\frac{x_i}{\operatorname{rms}} + \beta
$$

| Symbol | Meaning |
|---|---|
| $N$ | Vector length |
| $x_i$ | Input value |
| $\varepsilon$ | Stability constant added inside the root |
| $\operatorname{rms}$ | Root mean square of the input |
| $\gamma,\ \beta$ | Scalar scale and shift |
| $y_i$ | Output value |

The output depends on a **global** statistic of the whole vector. The
computation is therefore a reduction followed by a broadcast elementwise pass.

## Approach

Three kernels in one stream:

1. **`sumSquares`** (≤ 512 blocks): grid-stride `fmaf(x, x, acc)` in float32
   per thread, then a float64 block reduction into `g_partials[b]`.
2. **`computeInvRms`** (1 block): sums the partials in float64 and stores
   $1/\sqrt{\text{sum}/N + \varepsilon}$ in the `__device__` variable
   `g_inv_rms`. Storing the **reciprocal** turns $N$ divisions into $N$
   multiplications.
3. **`scaleShift`**: grid-stride `y = γ·(x·inv_rms) + β`. The multiplication
   order matches the reference's `gamma * (input / rms) + beta` up to one
   rounding, well within `1e-5`.

The scalar is handed from kernel to kernel through device memory, so there
is no host round trip.

## Cost Analysis

$$
Q = 4N + 8N = 12N\ \text{bytes}, \qquad W \approx 5N
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes: read $x$ (pass 1), read $x$ and write $y$ (pass 3) |
| $W$ | FLOPs (FMA for the squares, plus a multiply and an FMA in the output pass) |

$N = 10^5$ is only 400 KB, which is L2-resident and launch-latency-bound
(three kernels ≈ 10 µs). For a batch of many rows, as in real LLM layers, a
single kernel with one block (or warp) **per row** does the reduction and the
scaling without leaving the SM (see [Fused Residual + RMSNorm](../083-fused-residual-add-rms-norm/)).

## Pitfalls

- **Mean subtraction.** RMSNorm has none. Confusing it with LayerNorm fails
  immediately.
- **$\varepsilon$ placement.** It goes inside the square root, added to the
  mean of squares.
- **float32 sum of $10^5$ squares up to $10^4$.** The float64 upper levels
  keep the relative error around $10^{-12}$.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-5`,
including $N = 1$.

## Related

- [Fused Residual Add + RMSNorm](../083-fused-residual-add-rms-norm/), [Layer Norm](../113-layer-normalization/),
  [Batch Norm](../040-batch-normalization/). Tensara [RMS Norm](../../tensara/rms-norm/).

---
title: Weight Dequantization
platform: LeetGPU
upstream: medium/64_weight_dequantization
url: https://leetgpu.com/challenges/weight-dequantization
difficulty: medium
tags: [elementwise, quantization, block-scaling]
status: solved
---

# Weight Dequantization

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/weight-dequantization)

## Problem

Dequantise an $M \times N$ weight matrix whose scales are stored per
$T \times T$ **tile** ($M, N \le 8192$, $T \in \{16, 32, 64, 128\}$;
benchmark $M = N = 8192$, $T = 128$; tolerance `1e-5`). Block-wise scaling
is how modern low-precision formats (DeepSeek-V3's FP8 weights, MX formats)
keep quantisation error local. The kernel is the "unpack" step that runs
before, or fused into, a GEMM.

## Formulation

$$
Y_{ij} = X_{ij}\cdot S_{\lfloor i/T\rfloor,\ \lfloor j/T\rfloor}, \qquad S \in \mathbb R^{\lceil M/T\rceil \times \lceil N/T\rceil}
$$

| Symbol | Meaning |
|---|---|
| $M,\ N$ | matrix rows and columns |
| $T$ | tile size (`TILE_SIZE`) |
| $X_{ij}$ | quantised value (given as float32 here) |
| $S_{rc}$ | scale of tile $(r, c)$; row-major with $\lceil N/T\rceil$ columns |
| $Y_{ij}$ | dequantised value |
| $\lfloor i/T\rfloor$ | tile row of element row $i$ (edge tiles may be partial) |

## Approach

- 2-D grid of $64 \times 4$ thread blocks. `threadIdx.x` runs along the
  columns, so loads of $X$ and stores of $Y$ are coalesced 256-byte rows.
- Each thread computes its tile coordinates with two integer divisions and
  multiplies by the scale.
- **Scale reuse.** For $T = 128$, all 64 threads of a block row share one or
  two scales, and $T^2 = 16\,384$ elements share each scale overall. The
  scale matrix ($64 \times 64$ floats = 16 KB) stays in L1/L2 permanently.

## Cost analysis

$$
Q \approx 8MN + 4\left\lceil\frac{M}{T}\right\rceil\left\lceil\frac{N}{T}\right\rceil \ \text{bytes}, \qquad T_{\min} \approx \frac{8MN}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes: read $X$ and write $Y$ (4 bytes each); the scales are negligible |
| $\beta$ | DRAM bandwidth |

Benchmark: 537 MB, i.e. ≈ 270 µs at 2 TB/s. In a real int8/fp8 pipeline the
quantised $X$ would be 1 byte per element, and the multiply would happen in
the GEMM's register file without ever writing $Y$.

## Pitfalls

- **Partial edge tiles.** $\lceil N/T\rceil$ (not $N/T$) scale columns are
  used for the row stride of $S$.
- **Integer division cost.** It is hidden by memory latency. For power-of-two
  $T$ a shift would do.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md), for all four
tile sizes and non-multiple dimensions.

## Related

- [INT8 Quantized MatMul](../032-int8-quantized-matmul/), [INT4 MatMul](../081-int4-matmul/).
- Tensara [MXFP8 Dequantize](../../tensara/mxfp8-dequantize/), [NVFP4 Dequantize](../../tensara/nvfp4-dequantize/).

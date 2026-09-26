---
title: INT8 Quantized MatMul
platform: LeetGPU
upstream: medium/32_int8_quantized_matmul
url: https://leetgpu.com/challenges/int8-quantized-matmul
difficulty: medium
tags: [gemm, int8, quantization, tensor-cores, wmma]
status: solved
---

# INT8 Quantized MatMul

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/int8-quantized-matmul)

## Problem

Quantised matrix multiplication. $A$ ($M\times K$) and $B$ ($K \times N$) are
int8 with scales $s_A, s_B$ and zero points $z_A, z_B$. The int8 output $C$
has scale $s_C$ and zero point $z_C$ ($1 \le M, N, K \le 4096$; benchmark
$M = 8192$, $N = 4096$, $K = 2048$). The check is **bit-exact**. This is how
int8 inference works: integer tensor-core matmuls, followed by a float
"requantisation" epilogue.

## Formulation

$$
C_{ij} = \operatorname{clamp}\!\Bigl(\operatorname{rne}\bigl(\operatorname{fl}\bigl(\operatorname{fl}(\operatorname{fl}(S_{ij}\, s_A)\, s_B) / s_C\bigr)\bigr) + z_C,\ -128,\ 127\Bigr),
\qquad S_{ij} = \sum_{k=0}^{K-1} (A_{ik} - z_A)(B_{kj} - z_B)
$$

| Symbol | Meaning |
|---|---|
| $M,\ N,\ K$ | output rows, output columns, inner dimension |
| $A_{ik},\ B_{kj}$ | int8 input values |
| $z_A,\ z_B,\ z_C$ | zero points (integers in $[-128, 127]$) |
| $s_A,\ s_B,\ s_C$ | positive float32 scales |
| $S_{ij}$ | exact integer dot product of the zero-shifted inputs (int32) |
| $\operatorname{fl}(\cdot)$ | a single float32 operation, rounded to nearest; the order $((S\,s_A)\,s_B)/s_C$ is the reference's |
| rne | round to nearest, ties to even (`torch.round`, `rintf`) |
| clamp | saturate to the int8 range |

### Folding Out the Zero Points

$A_{ik} - z_A$ ranges over $[-255, 255]$ and no longer fits in int8, so the
shifted values cannot go directly to the int8 tensor cores. Expanding the
product instead:

$$
S_{ij} = \underbrace{\sum_k A_{ik}B_{kj}}_{P_{ij}\ (\text{int8 MMA})} - z_B \underbrace{\sum_k A_{ik}}_{r_i} - z_A \underbrace{\sum_k B_{kj}}_{c_j} + K z_A z_B
$$

| Symbol | Meaning |
|---|---|
| $P_{ij}$ | raw int8 × int8 product, accumulated in int32 on tensor cores |
| $r_i$ | row sum of $A$ (one int per row) |
| $c_j$ | column sum of $B$ (one int per column) |
| $K z_A z_B$ | constant correction term |

All terms are exact int32 arithmetic. $\lvert P\rvert \le 4096 \cdot 128^2 \approx 6.7\times10^7 < 2^{31}$,
so nothing overflows.

## Approach

1. **`rowSums`** (warp per row with a shuffle reduction) and **`colSums`**
   (thread per column, coalesced across columns) compute $r$ and $c$.
2. **`imma`**, the main GEMM:
   - Block = 4 warps computes a 64 × 64 tile of $C$. Each warp computes
     32 × 32, i.e. 2 × 2 WMMA `16x16x16` fragments with
     `signed char × signed char → int`.
   - K is walked in slices of 32. The int8 tiles are staged in shared memory
     as **contiguous 16 × 16 blocks**
     (`a_s[4][2][16][16]`, `b_s[2][4][16][16]`, `ldm = 16`).
   - Accumulators are stored to a shared int32 tile. The epilogue computes
     $S_{ij} = P_{ij} - z_B r_i - z_A c_j + K z_A z_B$, then requantises with
     *exactly* the reference's float32 operation order and `rintf`, adds
     $z_C$, clamps, and stores int8.

### Why Contiguous 16 × 16 Blocks?

WMMA requires fragment pointers aligned to 32 bytes. With 1-byte elements, a
row-major tile `a_s[64][32+pad]` puts the second fragment of a row at
column 16, i.e. a 16-byte offset, which is misaligned. Storing each 16 × 16
fragment as its own contiguous 256-byte block makes every fragment start on a
256-byte boundary. The [cuemu](../../tools/cuemu/README.md) emulator's
alignment check caught the original padded layout.

## Cost Analysis

$$
W = 2MNK \ \text{int ops}, \qquad Q \approx MK\frac{N}{64} + KN\frac{M}{64} + MN \ \text{bytes}
$$

| Symbol | Meaning |
|---|---|
| $W$ | integer multiply-adds × 2 |
| $Q$ | bytes (1 byte per int8 element) with 64 × 64 tiling |

Benchmark: $W \approx 1.4\times10^{11}$ ops. At ~600 TOPS of int8 tensor
throughput (A100) that is 0.23 ms of compute. Int8 halves the bytes of fp16
and doubles the tensor-core rate, which is why quantised inference is fast.
The 64 × 64 tile with synchronous staging reaches only a part of it.

## Pitfalls

- **Rounding mode.** `roundf` rounds half away from zero, while
  `torch.round` rounds half to even. `rintf` matches the reference.
- **Operation order.** `S * (sA*sB/sC)` is mathematically equal, but it
  rounds differently in float32 and fails the exact check.
- **Pre-pass race.** The row and column sums are separate kernels on the same
  stream, so they finish before the GEMM reads them.

## Verification

Bit-exact on all LeetGPU cases in [cuemu](../../tools/cuemu/README.md),
which emulates int8 WMMA including its alignment rules. Tests include
non-multiple-of-16 shapes and extreme zero points.

## Related

- [GEMM (fp16)](../022-gemm/), [INT4 MatMul](../081-int4-matmul/), [Weight Dequantization](../064-weight-dequantization/).
- Tensara [MXFP8 GEMM](../../tensara/mxfp8-gemm/), [NVFP4 GEMM](../../tensara/nvfp4-gemm/).

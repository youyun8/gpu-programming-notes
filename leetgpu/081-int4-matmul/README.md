---
title: INT4 Weight-Only Quantized MatMul
platform: LeetGPU
upstream: medium/81_int4_matmul
url: https://leetgpu.com/challenges/int4-weight-only-quantized-matmul
difficulty: medium
tags: [gemm, int4, quantization, tensor-cores, wmma, w4a16]
status: solved
---

# INT4 Weight-Only Quantized MatMul

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/int4-weight-only-quantized-matmul)

## Problem

Weight-only INT4 GEMM ("**W4A16**"), the core of GPTQ/AWQ-style LLM
inference. $x$ is an fp16 activation matrix ($M\times K$). $W$ ($N\times K$)
is stored as packed 4-bit integers, two per byte, with one fp16 scale per
group of $g$ consecutive weights along $K$. Compute $y = xW^{\mathsf T}$ in
fp16 ($M, N, K \le 8192$, $g \in \{2..128\}$; benchmark $4096^3$, $g = 128$;
tolerance `1e-2`).

## Formulation

$$
W_{nk} = \bigl(q_{nk} - 8\bigr)\cdot s_{n,\lfloor k/g\rfloor}, \qquad
y_{mn} = \operatorname{fp16}\!\Bigl(\sum_{k=0}^{K-1} x_{mk}\, W_{nk}\Bigr)
$$

$$
q_{n,2i} = \bigl\lfloor b_{ni} / 16 \bigr\rfloor \ (\text{high nibble}), \qquad q_{n,2i+1} = b_{ni} \bmod 16 \ (\text{low nibble})
$$

| Symbol | Meaning |
|---|---|
| $M,\ N,\ K$ | tokens (rows of $x$), output features, input features |
| $x_{mk}$ | fp16 activation |
| $b_{ni}$ | packed byte $i$ of weight row $n$ (`w_q`, shape $N \times K/2$) |
| $q_{nk}$ | unsigned 4-bit code, $0..15$ |
| $q - 8$ | signed weight in $[-8, 7]$ (offset encoding) |
| $g$ | quantisation group size along $K$ |
| $s_{n,j}$ | fp16 scale of group $j$ in row $n$ (shape $N\times K/g$) |
| $W_{nk}$ | dequantised weight |
| $y_{mn}$ | fp16 output |

### Why weight-only quantisation works

During LLM decoding, $M$ is small (a few tokens) and the GEMM is limited by
**reading the weights**. INT4 weights are 4× smaller than fp16, so
memory-bound decode gets up to 4× faster, while the activations and the math
stay in fp16. The group scales keep the quantisation error local: each group
of 128 weights has its own dynamic range.

## Approach

The tensor-core GEMM of [GEMM (fp16)](../022-gemm/) with **dequantisation
fused into the shared-memory staging**:

1. **Stage $x$**: a $64\times32$ fp16 tile, zero-padded, as usual.
2. **Stage $W$**: each thread reads **one packed byte**, i.e. two
   consecutive weights of one row. It unpacks the nibbles, subtracts 8, looks
   up the two group scales (they may differ when $g = 2$ and the pair
   straddles a group boundary), multiplies, and writes two fp16 values into
   `w_s[n][k]`. $W$ never exists in dequantised form in global memory.
3. **MMA**: `w_s` holds $W$ as $[n][k]$, i.e. $W^{\mathsf T}$ is
   column-major. It is loaded as the B fragment with **`wmma::col_major`**,
   so no transpose is needed. Each warp computes 2 × 2 fragments with fp32
   accumulators.
4. **Epilogue**: shared float tile → fp16 stores with bounds checks.

## Cost analysis

$$
W_{\text{flop}} = 2MNK, \qquad
Q_W = \frac{NK}{2} + 2\frac{NK}{g}\ \text{bytes}\ (\text{vs. } 2NK \text{ for fp16}), \qquad
Q_x = 2MK
$$

| Symbol | Meaning |
|---|---|
| $W_{\text{flop}}$ | tensor-core FLOPs |
| $Q_W$ | bytes to read the weights once: 0.5 byte per weight plus the scales |
| $Q_x$ | bytes to read the activations once |

At the benchmark ($4096^3$, $g = 128$): $Q_W = 8.4$ MB versus 33.5 MB for fp16,
and $W_{\text{flop}} = 137$ GFLOP, so this large-$M$ case is compute-bound. The
memory saving pays off at small $M$ (decode), where the same kernel is
bandwidth-bound on $Q_W$. There, a GEMV-style kernel (one warp per output
column) would be the better tool.

## Pitfalls

- **Nibble order.** The **high** nibble holds the *even* index $2i$. That is
  the reverse of the more common low-first packing (compare the Tensara FP4
  problems).
- **Offset encoding.** The signed value is $q - 8$, not a two's-complement nibble.
- **Group boundary inside a byte pair** for $g = 2$: fetch each weight's scale
  separately.
- **fp16 rounding** of the dequantised weights before the MMA adds error
  compared with the reference's float32 dequantisation, but it stays well
  inside `1e-2`.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-2`,
for every group size, with $M, N$ not multiples of 64.

## Related

- [INT8 Quantized MatMul](../032-int8-quantized-matmul/), [Weight Dequantization](../064-weight-dequantization/),
  [GEMM (fp16)](../022-gemm/). Tensara [NVFP4 GEMM](../../tensara/nvfp4-gemm/), [MXFP4 GEMM](../../tensara/mxfp4-gemm/).

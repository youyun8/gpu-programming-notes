---
title: General Matrix Multiplication (GEMM)
platform: LeetGPU
upstream: medium/22_gemm
url: https://leetgpu.com/challenges/general-matrix-multiplication-gemm
difficulty: medium
tags: [gemm, fp16, tensor-cores, wmma]
status: solved
---

# General Matrix Multiplication (GEMM)

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/general-matrix-multiplication-gemm)

## Problem

Half-precision GEMM with scaling: $C \leftarrow \alpha AB + \beta C$, where
$A$ is $M \times K$, $B$ is $K \times N$, and $C$ is $M \times N$, all fp16
row-major, with float32 scalars $\alpha, \beta$ ($16 \le M, N, K \le 4096$,
not necessarily multiples of 16; benchmark $M = N = K = 1024$). Accumulation
must be float32 and the tolerance is `0.05`. This is the problem that
introduces **tensor cores**: dedicated matrix units that execute a small
matrix multiply-accumulate per warp instruction, at many times the fp32 FMA
throughput.

## Formulation

$$
C_{rc} \leftarrow \operatorname{fp16}\!\left(\alpha \sum_{k=0}^{K-1} \operatorname{fp32}(A_{rk})\operatorname{fp32}(B_{kc}) \;+\; \beta\,\operatorname{fp32}(C^{\text{old}}_{rc})\right)
$$

| Symbol | Meaning |
|---|---|
| $M,\ N,\ K$ | rows of $A$/$C$, columns of $B$/$C$, and the inner dimension |
| $A_{rk},\ B_{kc}$ | fp16 inputs |
| $C^{\text{old}}_{rc}$ | initial content of $C$ (fp16) |
| $\alpha,\ \beta$ | float32 scalars |
| fp32(·), fp16(·) | conversion to float32, and round-to-nearest conversion back to fp16 |

### Tensor-core fragments (WMMA 16 × 16 × 16)

One `wmma::mma_sync` call per warp computes

$$
D = A_f B_f + C_f, \qquad A_f \in \mathrm{fp16}^{16\times16},\ B_f \in \mathrm{fp16}^{16\times16},\ C_f, D \in \mathrm{fp32}^{16\times16}
$$

| Symbol | Meaning |
|---|---|
| $A_f,\ B_f$ | matrix fragments loaded from shared memory with `load_matrix_sync` |
| $C_f,\ D$ | accumulator fragment (float32), kept in registers across the whole $K$ loop |

These are 4096 multiply-adds per warp instruction. The register layout of a
fragment is opaque (architecture-specific), so the API only allows
load/store/fill/mma on it. (Chapter [05](../../tutorials/05-amd-cdna3-mfma.md)
contrasts this with AMD MFMA, where the layout is documented.)

## Approach

### Tiling hierarchy

| Level | Tile of $C$ | Notes |
|---|---|---|
| Block (4 warps) | 64 × 64 | grid $\lceil N/64\rceil \times \lceil M/64\rceil$ |
| Warp | 32 × 32 | 2 × 2 accumulator fragments |
| MMA | 16 × 16 × 16 | one `mma_sync` |

### Main loop over $K$ in steps of 32

1. Stage `a_s[64][32+8]` and `b_s[32][64+8]` (fp16) from global memory,
   **zero-filling** out-of-range elements. $M$, $N$, $K$ then do not need to
   be multiples of 16, and padding with zeros adds nothing to the dot
   products.
2. `__syncthreads()`.
3. For `kk = 0, 16`, each warp loads 2 A-fragments and 2 B-fragments and
   issues 4 `mma_sync`. Each A fragment is reused for 2 MMAs, and each B
   fragment for 2.
4. `__syncthreads()`.

### Epilogue

Accumulators are stored to a shared float32 tile `c_s[64][64+4]`. Then every
thread handles elements of the tile: it reads $C^{\text{old}}$, computes
$\alpha\cdot\text{acc} + \beta C^{\text{old}}$ in float32, converts to fp16,
and stores with a bounds check. Going through shared memory is what makes the
per-element $\beta$ term and the bounds checks possible, since fragments
cannot be indexed element by element portably.

### Alignment rules (why the odd pitches)

WMMA requires the pointer to be 32-byte aligned and the leading dimension
`ldm` to be a multiple of 16 bytes: 8 halves or 4 floats. Pitches of 40 and
72 halves and 68 floats satisfy that. They also shift consecutive rows by
16 bytes relative to the 128-byte bank cycle, which spreads the fragment
loads over the banks. All shared arrays are declared `__align__(32)`.

## Cost analysis

$$
W = 2MNK, \qquad
Q \approx 2\left(MK\,\frac{N}{64} + KN\,\frac{M}{64}\right) + 4MN, \qquad
I \approx \frac{2MNK}{2\cdot 2MNK/64} = 32\ \text{FLOP/byte}
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs |
| $Q$ | DRAM bytes: fp16 $A$ re-read once per column-tile, $B$ once per row-tile, and $C$ read and written (2 + 2 bytes) |
| $I$ | arithmetic intensity, ignoring the $C$ term |

At $1024^3$, $W \approx 2.1$ GFLOP. On an A100 (312 TFLOP/s fp16 tensor)
the compute floor is ≈ 7 µs. A 64 × 64 block tile with synchronous
shared-memory staging reaches only a fraction of that. Production kernels use
128 × 128+ tiles, `cp.async`/TMA multi-stage pipelines and `ldmatrix`-based
fragment loads (CUTLASS, cuBLAS).

## Pitfalls

- **Misaligned WMMA pointers** fault or read garbage on real hardware. The
  [cuemu](../../tools/cuemu/README.md) emulator checks both alignment rules,
  and it caught a violation during development.
- **In-place $C$.** $\beta$ must use the *original* $C$. Each thread reads
  `c[idx]` just before overwriting that same element, so no other thread
  depends on it.
- **fp16 overflow.** $\lvert x\rvert > 65504$ becomes $\infty$ in the final
  conversion. Accumulating in fp32 avoids intermediate overflow.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) with the WMMA
emulation, including $M, N, K$ not divisible by 16. The kernel also compiles
with `nvcc -arch=sm_80`.

## Related

- [Matrix Multiplication (fp32)](../002-matrix-multiplication/), [FP16 Batched MatMul](../057-fp16-batched-matmul/),
  [INT8 MatMul](../032-int8-quantized-matmul/), [INT4 MatMul](../081-int4-matmul/).
- The AMD equivalent, MFMA: [tutorial 05](../../tutorials/05-amd-cdna3-mfma.md).

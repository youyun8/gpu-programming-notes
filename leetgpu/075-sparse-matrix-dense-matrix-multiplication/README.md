---
title: Sparse Matrix-Dense Matrix Multiplication
platform: LeetGPU
upstream: medium/75_sparse_matrix_dense_matrix_multiplication
url: https://leetgpu.com/challenges/sparse-matrix-dense-matrix-multiplication
difficulty: medium
tags: [gemm, sparsity, register-blocking]
status: solved
---

# Sparse Matrix-Dense Matrix Multiplication

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/sparse-matrix-dense-matrix-multiplication)

## Problem

$C = AB$ where $A$ ($M\times N$) is 60–70% zeros but stored densely, and
$B$ ($N \times K$) is dense, all float32 row-major ($M, N, K \le 8192$;
benchmark $M = 4096$, $N = 2048$, $K = 512$; tolerance `1e-3`). Is exploiting
the sparsity worth it? At this density, **no**. The page explains why, with
numbers.

## Formulation

$$
C_{ij} = \sum_{k=0}^{N-1} A_{ik} B_{kj} = \sum_{k\,:\,A_{ik}\neq 0} A_{ik}B_{kj}
$$

| Symbol | Meaning |
|---|---|
| $M,\ N,\ K$ | rows of $A$, inner dimension, columns of $B$ |
| $A_{ik}$ | sparse matrix element (mostly zero) |
| $B_{kj}$ | dense matrix element |
| $C_{ij}$ | output |
| nnz | number of non-zeros of $A$, ≈ $0.35MN$ |

### Dense vs. Sparse Break-Even

$$
W_{\text{dense}} = 2MNK, \qquad W_{\text{sparse}} = 2\,\text{nnz}\cdot K = 2\rho MNK
$$

| Symbol | Meaning |
|---|---|
| $W_{\text{dense}}$ | FLOPs of a dense GEMM |
| $W_{\text{sparse}}$ | useful FLOPs if only non-zeros are processed |
| $\rho$ | density nnz / $(MN)$, here ≈ 0.3–0.4 |

A sparse kernel saves at most $1/\rho \approx 3\times$ in FLOPs, but:

- It first needs a CSR conversion: one pass over $A$ plus a scan (see
  [Stream Compaction](../072-stream-compaction/)).
- Each non-zero triggers a *gather* of row $k$ of $B$. This is irregular, and
  it loses the register and shared-memory reuse that make dense GEMM run at a
  high fraction of peak.
- Dense tiled GEMM runs at 50–90% of peak, while CSR SpMM with this density
  typically reaches 5–20%.

The crossover on GPUs is typically at $\rho \approx 1$–$5\%$. At 35% the
dense kernel wins, which is also what cuSPARSE's own guidance says.

## Approach

The 64 × 64 register-blocked SGEMM of [Matrix Multiplication](../002-matrix-multiplication/):
256 threads, 16-wide $K$ slices of $A$ (stored transposed) and $B$ in shared
memory, 4 × 4 strided outputs per thread, zero-padded edge tiles. The
multiplication by zeros is simply done; it is free compared with the
alternative.

## Cost Analysis

$$
W = 2MNK, \qquad Q \approx 4\left(MN\frac{K}{64} + NK\frac{M}{64} + MK\right)
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs of the dense product |
| $Q$ | DRAM bytes with 64 × 64 tiling |

Benchmark: $W = 8.6$ GFLOP, about 0.5 ms on a ~20 TFLOP/s fp32 GPU at
realistic efficiency.

## Pitfalls

- **"Sparse" in the name.** Measure before assuming a sparse format helps.
- **Dimension naming.** $N$ is the inner dimension here.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-3`.

## Related

- [Sparse Matrix-Vector](../018-sparse-matrix-vector-multiplication/) (the same argument for GEMV),
  [Matrix Multiplication](../002-matrix-multiplication/).

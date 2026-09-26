---
title: Sparse Matrix-Vector Multiplication
platform: LeetGPU
upstream: medium/18_sparse_matrix_vector_multiplication
url: https://leetgpu.com/challenges/sparse-matrix-vector-multiplication
difficulty: medium
tags: [gemv, warp-per-row, memory-bound, sparse]
status: solved
---

# Sparse Matrix-Vector Multiplication

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/sparse-matrix-vector-multiplication)

## Problem

Compute $\mathbf y = A\mathbf x$ where $A$ is an $M \times N$ matrix with
about 60–70% zeros, but **stored densely** in row-major order
($1 \le M, N \le 10^4$; benchmark $M = 1000$, $N = 10\,000$; tolerance
`1e-3`). The count `nnz` is provided but no index structure (CSR/COO) is. The
lesson is to look at what actually limits a kernel before optimising: here
it is the bytes, not the FLOPs.

## Formulation

$$
y_r = \sum_{c=0}^{N-1} A_{rc}\, x_c, \qquad 0 \le r < M
$$

| Symbol | Meaning |
|---|---|
| $M,\ N$ | Rows and columns of $A$ |
| $A_{rc}$ | Matrix element at offset $rN + c$ (often zero) |
| $x_c$ | Dense input vector, length $N$ |
| $y_r$ | Output vector, length $M$ |
| nnz | Number of non-zero elements of $A$ (unused by the kernel) |

## Approach

### Why "Sparse" Does Not Help Here

Every element of $A$, zero or not, must be read to know whether it is zero,
so the traffic is $4MN$ bytes regardless of sparsity. Skipping zeros would
only save FMAs, which are not the bottleneck. Converting to CSR on the GPU
first would add a full pass over $A$ plus a scan, and only pays off when the
same matrix is multiplied many times.

### Warp per Row

- 8 warps per block. Warp $w$ handles row $r$.
- Lane $\ell$ accumulates $\sum_{c \equiv \ell \pmod{32}} A_{rc} x_c$ with
  `fmaf`. For each step, the 32 lanes read 32 consecutive floats of row $r$:
  128 bytes, perfectly coalesced.
- $\mathbf x$ is at most 40 KB, is read through `__ldg` (read-only cache),
  and stays resident in L1/L2 across rows.
- A 5-step `__shfl_down_sync` reduction gives $y_r$ in lane 0.

One *thread* per row would be the alternative. Adjacent threads would then
read addresses $N$ floats apart, and every load would touch a different
sector: a 32× waste of bandwidth.

## Cost Analysis

$$
Q \approx 4MN + 4N + 4M, \qquad W = 2MN, \qquad I \approx \frac{2MN}{4MN} = \frac12, \qquad T_{\min} \approx \frac{4MN}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | Bytes: the whole matrix once, $\mathbf x$ once (cached), $\mathbf y$ once |
| $W$ | FLOPs |
| $I$ | Arithmetic intensity |
| $\beta$ | DRAM bandwidth |

Benchmark: $4MN = 40$ MB, so $T_{\min} \approx 20\ \mu s$ at 2 TB/s. With only
1000 rows = 1000 warps, occupancy is modest. Splitting long rows across
several warps (and reducing via shared memory) can help on large GPUs.

## Pitfalls

- **Returning inside the warp.** `if (row >= m) return;` is safe because the
  whole warp shares the same `row`, so the full-mask shuffle is never
  executed with missing lanes.
- **Tolerance.** `1e-3` reflects a 10 000-term float32 dot product; the
  shuffle tree keeps the error well below it.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md), including
$M = 1$ and $N < 32$, where most lanes are idle.

## Related

- [Dot Product](../017-dot-product/), [Sparse × Dense MatMul](../075-sparse-matrix-dense-matrix-multiplication/).
- Tensara [Matrix-Vector](../../tensara/matrix-vector/), [NVFP4 GEMV](../../tensara/nvfp4-gemv/).

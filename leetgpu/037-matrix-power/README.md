---
title: Matrix Power
platform: LeetGPU
upstream: medium/37_matrix_power
url: https://leetgpu.com/challenges/matrix-power
difficulty: medium
tags: [gemm, binary-exponentiation, linear-algebra]
status: solved
---

# Matrix Power

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/matrix-power)

## Problem

Compute $A^P$ for an $N \times N$ float32 matrix ($1 \le N \le 1024$,
$1 \le P \le 20$, $\lvert A_{ij}\rvert \le 10$; benchmark $N = 512$; tolerance
`1e-4`). The naive approach does $P - 1$ matrix products.
**Exponentiation by squaring** needs only $O(\log P)$. The ordering of the
products also affects float32 rounding, and it is chosen to match
`torch.linalg.matrix_power`.

## Formulation

Write $P$ in binary, $P = \sum_{j} b_j 2^j$ with $b_j \in \{0, 1\}$. Then

$$
A^P = \prod_{j\,:\,b_j = 1} Z_j, \qquad Z_0 = A,\quad Z_{j+1} = Z_j^2
$$

| Symbol | Meaning |
|---|---|
| $A$ | input matrix, $N \times N$ |
| $P$ | exponent, $\ge 1$ |
| $b_j$ | $j$-th bit of $P$ |
| $Z_j$ | $A^{2^j}$, obtained by repeated squaring |

The number of products is

$$
\#\text{GEMM} = \underbrace{\lfloor\log_2 P\rfloor}_{\text{squarings}} + \underbrace{\operatorname{popcount}(P) - 1}_{\text{multiplies into the result}} \;\le\; 2\lfloor\log_2 P\rfloor
$$

| Symbol | Meaning |
|---|---|
| popcount$(P)$ | number of set bits of $P$ |

For $P = 20 = 10100_2$: 4 squarings + 1 multiply = **5 GEMMs** instead of 19.

### Matching PyTorch's Rounding

Matrix multiplication is associative mathematically but not in floating
point. With entries up to 10, $A^{20}$ has huge dynamic range, and a different
association order changes the result by more than `1e-4` relative. The
solution mirrors `matrix_power` exactly:

| $P$ | Order of products |
|---|---|
| 1 | $A$ (copy) |
| 2 | $A \cdot A$ |
| 3 | $(A \cdot A)\cdot A$ |
| ≥ 4 | walk bits from LSB. $Z \leftarrow Z^2$ after the first bit. On a set bit, $\text{res} \leftarrow \text{res}\cdot Z$ (the first set bit copies $Z$) |

## Approach

- Each product uses the 64 × 64 register-blocked SGEMM from
  [Matrix Multiplication](../002-matrix-multiplication/), a
  $\lceil N/64\rceil^2$ grid of 256 threads each.
- Four scratch buffers ($Z$, $Z_{\text{next}}$, result, result$_{\text{next}}$)
  are **ping-ponged** so that a GEMM never writes to one of its own inputs
  (in-place GEMM would be a race).
- All launches go into the same stream, so each GEMM sees the previous one's
  result without explicit synchronisation.

## Cost Analysis

$$
W = 2N^3 \cdot \#\text{GEMM}, \qquad Q \approx \#\text{GEMM}\cdot 4\left(2N^2\frac{N}{64} + N^2\right)
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs over all products |
| $Q$ | DRAM bytes (each GEMM re-reads its inputs once per tile row/column) |

$N = 512$, $P = 20$: $W = 5 \cdot 2.7\times10^8 = 1.3$ GFLOP, about 0.1–0.2 ms
of fp32 FMA on a modern GPU. The whole working set (4 MB) lives in L2.

## Pitfalls

- **Aliasing.** `matmul(z, z, z)` would overwrite $Z$ while other blocks are
  still reading it.
- **Order of multiplication.** Mathematically $\text{res}\cdot Z = Z \cdot \text{res}$
  here (powers of $A$ commute), but the float32 results differ. Keep the
  reference's order.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-4`,
for $P = 1..20$ including the special cases $P \le 3$.

## Related

- [Matrix Multiplication](../002-matrix-multiplication/), Tensara [Matrix Power](../../tensara/matrix-power/).

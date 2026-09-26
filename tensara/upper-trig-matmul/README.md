---
title: Upper Triangular Matrix Multiplication
platform: Tensara
upstream: upper-trig-matmul
url: https://tensara.org/problems/upper-trig-matmul
difficulty: medium
tags: [matmul, sgemm, triangular, work-skipping]
status: solved
---

# Upper Triangular Matrix Multiplication

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/upper-trig-matmul)

## Problem

Multiply two upper-triangular $N\times N$ FP32 matrices ($N$ = 2048 …
8192). The reference applies `torch.triu` to both and does a dense
matmul. The check is `rtol = 1e-4`, `atol = 1e-3`.

## Formulation

$$
U_{ij} = 0 \ \text{for } i > j, \qquad
C_{ij} = \begin{cases} \displaystyle\sum_{k=i}^{j} A_{ik}B_{kj}, & i \le j \\ 0, & i > j \end{cases}
$$

| Symbol | Meaning |
|---|---|
| $U$ | an upper-triangular matrix |
| $A, B$ | inputs after `triu` |
| $C$ | product, upper triangular |
| $k \in [i, j]$ | $A_{ik} \ne 0$ needs $k \ge i$, $B_{kj} \ne 0$ needs $k \le j$ |

For a tile with rows $[r_0, r_0+64)$ and columns $[c_0, c_0+64)$:

$$
k \in \bigl[\,r_0,\ \min(N,\ c_0 + 64)\,\bigr)
$$

| Symbol | Meaning |
|---|---|
| $r_0, c_0$ | first row and column of the tile |

## Approach

The mirror image of [Lower Triangular Matmul](../lower-trig-matmul/):
tiles strictly below the diagonal are zero-filled, the others loop $k$
over the range above, and loads mask $A_{ik}$ with $k < i$ and $B_{kj}$
with $k > j$.

### The shared SGEMM kernel

All matmul pages on Tensara use the same register-blocked FP32 kernel
(`gemmKernel<kTransB, Epi>`):

1. **Block tile $64\times64$**, 256 threads; each thread owns a $4\times4$
   patch of outputs at rows `ty + 16i`, columns `tx + 16j`. The stride-16
   layout makes every store instruction of a warp hit 16 consecutive
   columns (coalesced), and makes shared-memory reads conflict-free.
2. **K-slices of 16.** Per slice the block copies a $64\times16$ panel of
   $A$ (stored transposed as `a_tile[k][m]`) and a $16\times64$ panel of
   $B$ into shared memory (rows padded by 4 floats), then synchronizes.
3. **Inner product in registers.** For each of the 16 values of $k$, a
   thread loads 4 values of $A$ and 4 of $B$ from shared memory and does
   $4\times4 = 16$ FMAs (an outer product).
4. **Epilogue functor.** The accumulator goes through `epi(v, row, col)`
   before the single store. This is where bias, activations, scaling or
   elementwise multiplies are fused, so the product never makes a round
   trip through DRAM.
5. `kTransB = true` reads $B$ as $N\times K$ ("NT", the `nn.Linear`
   weight layout) and transposes it while staging.

Data reuse at each level of the hierarchy:

$$
I_{\text{L2}} = \frac{2\,T_M T_N T_K}{4\,T_K\,(T_M + T_N)} = \frac{T_M T_N}{2\,(T_M + T_N)} = 16\ \tfrac{\text{flop}}{\text{byte}}, \qquad
I_{\text{smem}} = \frac{2 \cdot r_M r_N}{4\,(r_M + r_N)} = 1\ \tfrac{\text{flop}}{\text{byte}}
$$

| Symbol | Meaning |
|---|---|
| $T_M, T_N, T_K$ | block tile: 64, 64, 16 |
| $r_M, r_N$ | per-thread register tile: 4 × 4 |
| $I_{\text{L2}}$ | flops per byte loaded from L2/DRAM into shared memory |
| $I_{\text{smem}}$ | flops per byte read from shared memory (16 FMAs per 8 loads) |

This reaches roughly 40–60 % of FP32 peak. The next steps are the ones
covered in the [SGEMM tutorial](../../tutorials/04-tiled-matmul.md):
$128\times128$ tiles with $8\times8$ per thread, `float4` shared loads,
double-buffered `cp.async` staging, and finally tensor cores (TF32) where
the tolerance allows.

## Cost analysis

$$
W_{\text{dense}} = 2N^3, \qquad
W_{\text{tri}} = 2\sum_{d=0}^{N-1} (N - d)(d + 1) \approx \frac{N^3}{3}, \qquad
\frac{W_{\text{tri}}}{W_{\text{dense}}} \approx \frac{1}{6}
$$

| Symbol | Meaning |
|---|---|
| $W_{\text{dense}}$ | flops of a dense $N\times N$ product |
| $W_{\text{tri}}$ | flops actually needed: output $(i, j)$ in the triangle with $d = \lvert i - j\rvert$ needs $d + 1$ FMAs, and there are $N - d$ such outputs |
| $d$ | distance from the diagonal |

With 64-wide tiles the kernel does a little more (whole tiles and
16-aligned $k$ ranges), close to the $1/6$ bound for $N \ge 2048$. At
$N = 8192$ that is ~0.18 TFLOP instead of 1.1.

## Pitfalls

- **Tighter tolerance** than the lower-triangular problem (`atol = 1e-3`);
  it passes because skipped terms are exact zeros.
- **Mask at load**, as for the lower case.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Lower Triangular Matmul](../lower-trig-matmul/), [Square Matmul](../square-matmul/).

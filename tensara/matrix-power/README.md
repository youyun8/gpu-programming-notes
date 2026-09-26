---
title: Matrix Nth Power
platform: Tensara
upstream: matrix-power
url: https://tensara.org/problems/matrix-power
difficulty: medium
tags: [matmul, sgemm, exponentiation-by-squaring]
status: solved
---

# Matrix Nth Power

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/matrix-power)

## Problem

Compute $A^P$ for a $512\times512$ FP32 matrix and $P \in \{2, 4, 8\}$
(the code handles any $P \ge 0$). The check is `rtol = 1e-4`,
`atol = 1e-3`, so the order of multiplications matters for matching the
reference's rounding.

## Formulation

$$
A^0 = I, \qquad A^{P} = \prod_{t\,:\,\beta_t = 1} A^{2^t}, \qquad P = \sum_{t} \beta_t\,2^t
$$

| Symbol | Meaning |
|---|---|
| $A$ | input matrix, $n\times n$ ($n = 512$) |
| $I$ | identity matrix |
| $P$ | the exponent |
| $\beta_t$ | bit $t$ of $P$ (0 or 1) |
| $A^{2^t}$ | obtained by repeated squaring: $A^{2^{t+1}} = A^{2^t}A^{2^t}$ |

Binary exponentiation needs

$$
\#\text{GEMMs} = \lfloor \log_2 P \rfloor + \operatorname{popcount}(P) - 1 \quad \text{instead of } P - 1
$$

| Symbol | Meaning |
|---|---|
| $\operatorname{popcount}(P)$ | number of set bits of $P$ |

For $P = 8$: 3 squarings instead of 7 products.

## Approach

1. **Special cases**: $P = 0$ writes the identity; $P = 1$ copies; $P = 2$
   is one product; $P = 3$ is $(AA)A$.
2. **General case**, following the order of `torch.linalg.matrix_power`:
   walk the bits of $P$ from the least significant; $z$ goes
   $A, A^2, A^4, \dots$ by squaring; for each set bit the running result is
   multiplied by $z$ (or initialised to $z$). Four scratch buffers (two for
   $z$, two for the result) are ping-ponged so no product writes into one
   of its own inputs.
3. Each product is the $64\times64$ register-blocked SGEMM described
   below (a local copy specialised for square matrices).

### The Shared SGEMM Kernel

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

## Cost Analysis

$$
W = 2n^3\cdot\#\text{GEMMs}, \qquad Q_{\min} = 4\,(2n^2)\ \text{bytes}, \qquad T_{\min} = \max\left(\frac{W}{F},\ \frac{Q_{\min}}{\beta}\right)
$$

| Symbol | Meaning |
|---|---|
| $W$ | floating-point operations (one FMA = 2 flops) |
| $Q_{\min}$ | compulsory DRAM bytes (each operand read once, output written once) |
| $F$ | FP32 peak (tens of TFLOP/s on current GPUs) |
| $\beta$ | DRAM bandwidth |
| $T_{\min}$ | roofline lower bound |

At $n = 512$ one GEMM is 0.27 GFLOP on 64 blocks, fewer blocks than SMs,
so each GEMM is latency-bound (tens of µs). The dependency chain is
serial; only fewer GEMMs (binary exponentiation) help.

## Pitfalls

- **Aliasing**: `matmul(z, z, z)` would overwrite $z$ while reading it;
  hence the ping-pong buffers.
- **Multiplication order**: $A^4 = (A^2)^2$ and $((AA)A)A$ round
  differently; mirroring PyTorch's algorithm keeps the error well inside
  the tolerance.
- **$P = 0$** returns $I$, not $A$.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Square Matmul](../square-matmul/), LeetGPU [Matrix Power](../../leetgpu/037-matrix-power/).

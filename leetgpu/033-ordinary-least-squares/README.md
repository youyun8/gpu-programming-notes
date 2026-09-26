---
title: Ordinary Least Squares
platform: LeetGPU
upstream: medium/33_ordinary_least_squares
url: https://leetgpu.com/challenges/ordinary-least-squares
difficulty: medium
tags: [linear-algebra, cholesky, normal-equations, fp64]
status: solved
---

# Ordinary Least Squares

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/ordinary-least-squares)

## Problem

Fit a linear model by least squares: given $X \in \mathbb R^{n\times f}$ and
$\mathbf y \in \mathbb R^n$ (float32; $n \le 10^5$, $f \le 1000$, $n \ge f$,
$X$ full rank; benchmark $n = f = 32$), return the coefficient vector
$\boldsymbol\beta$, with `atol = rtol = 1e-2`. The reference solves the
**normal equations** with a Cholesky factorisation. This problem is a small
dense linear-algebra pipeline on the GPU: a Gram matrix (GEMM-like),
a factorisation (sequential in $k$, parallel within each step), and two
triangular solves.

## Formulation

$$
\boldsymbol\beta = \arg\min_{\boldsymbol\beta} \lVert X\boldsymbol\beta - \mathbf y\rVert_2^2
\quad\Longleftrightarrow\quad
\underbrace{X^{\mathsf T}X}_{G}\ \boldsymbol\beta = \underbrace{X^{\mathsf T}\mathbf y}_{\mathbf b}
$$

$$
G = LL^{\mathsf T}, \qquad L\mathbf z = \mathbf b\ \ (\text{forward}), \qquad L^{\mathsf T}\boldsymbol\beta = \mathbf z\ \ (\text{backward})
$$

| Symbol | Meaning |
|---|---|
| $n$ | number of samples (rows of $X$) |
| $f$ | number of features (columns of $X$) |
| $X$ | feature matrix, row-major, element $X_{sj}$ at $sf + j$ |
| $\mathbf y$ | target vector |
| $\boldsymbol\beta$ | coefficients (output) |
| $\lVert\cdot\rVert_2$ | Euclidean norm |
| $G$ | Gram matrix $X^{\mathsf T}X$ ($f\times f$, symmetric positive definite for full-rank $X$) |
| $\mathbf b$ | right-hand side $X^{\mathsf T}\mathbf y$ |
| $L$ | lower-triangular Cholesky factor |
| $\mathbf z$ | intermediate vector of the forward solve |

### Right-Looking Cholesky, Step $k = 0 \dots f-1$

$$
L_{kk} = \sqrt{G_{kk}}, \qquad L_{ik} = \frac{G_{ik}}{L_{kk}}\ (i > k), \qquad G_{ij} \leftarrow G_{ij} - L_{ik}L_{jk}\ (j \le i,\ i, j > k)
$$

| Symbol | Meaning |
|---|---|
| $k$ | current pivot column |
| $G_{ij}$ | trailing sub-matrix, updated in place (lower triangle only) |
| $L_{ik}$ | column $k$ of $L$, overwriting $G$'s column |

## Approach

1. **`gramTiled`**: $G = X^{\mathsf T}X$ as a tiled "$A^{\mathsf T}A$" GEMM.
   A $16\times16$ block computes a $16\times16$ tile of $G$. It walks the
   sample dimension in slices of 16, staging $X[s_0{:}s_0{+}16,\ i_0{:}i_0{+}16]$
   and $X[s_0{:}s_0{+}16,\ j_0{:}j_0{+}16]$ in shared memory (padded to 17
   against bank conflicts), and accumulates in **float64**.
2. **`xtY`**: one thread per feature $j$ loops over samples. Consecutive
   threads read consecutive $X_{sj}$, which is coalesced.
3. **`choleskySolve`** (one 1024-thread block):
   - For each $k$: thread 0 computes the pivot, then all threads scale
     column $k$, then all threads apply the rank-1 trailing update (flattened
     over $(i, j)$). There is a barrier after each phase.
   - Forward and backward substitution: for each row, a block-wide reduction
     computes $\sum_j L_{ij}z_j$.

### Why float64

The condition number squares when forming the normal equations:

$$
\kappa_2(X^{\mathsf T}X) = \kappa_2(X)^2
$$

| Symbol | Meaning |
|---|---|
| $\kappa_2(\cdot)$ | 2-norm condition number (ratio of the largest to the smallest singular value) |

With inputs up to $\pm1000$ and random features, $\kappa(X)$ of $10^2$–$10^3$
gives $\kappa(G)$ of $10^4$–$10^6$. That exhausts float32's $\sim10^{-7}$
precision, but not float64's $\sim10^{-16}$.

## Cost Analysis

$$
W_G = 2nf^2, \qquad W_{\text{chol}} \approx \frac{f^3}{3}, \qquad W_{\text{solve}} = 2f^2, \qquad \text{sync steps} = O(f)
$$

| Symbol | Meaning |
|---|---|
| $W_G$ | FLOPs to form $G$ (only half is needed by symmetry; both halves are computed) |
| $W_{\text{chol}}$ | Cholesky FLOPs |
| $W_{\text{solve}}$ | two triangular solves |
| sync steps | the factorisation is inherently sequential over $k$, with ~3 barriers per step |

For $n = 10^5$, $f = 1000$: $W_G = 2\times10^{11}$ (float64), which dominates.
For the benchmark ($32 \times 32$) everything is latency. A single block with
$O(f)$ barriers is the right tool for small $f$. Large $f$ would call for a
blocked, multi-block Cholesky (as in cuSOLVER).

## Pitfalls

- **Float32 Gram matrix.** It can fail the `1e-2` tolerance on
  ill-conditioned inputs.
- **Upper triangle.** Only the lower triangle is updated and read. The upper
  half of `gram` holds stale values and is never touched by the solves.
- **Temporary buffers** ($f^2 + 2f$ doubles) are allocated per call and freed
  after synchronising.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-2`,
including $n = f$ (square systems) and $f = 1$.

## Related

- [Logistic Regression](../034-logistic-regression/) (the same Gram + Cholesky inside Newton's method).

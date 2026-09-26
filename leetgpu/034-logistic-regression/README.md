---
title: Logistic Regression
platform: LeetGPU
upstream: medium/34_logistic_regression
url: https://leetgpu.com/challenges/logistic-regression
difficulty: medium
tags: [optimization, newton, irls, cholesky, fp64]
status: solved
---

# Logistic Regression

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/logistic-regression)

## Problem

Fit binary logistic regression: $X \in \mathbb R^{n\times f}$ and labels
$y_i \in \{0, 1\}$ ($n \le 10^5$, $f \le 1000$; benchmark $n = 16$, $f = 8$;
tolerance `1e-2`). The reference runs **Newton–Raphson (IRLS)** with a tiny
L2 term until the step norm drops below $10^{-8}$. Matching its fixed point
within `1e-2` essentially requires running the same algorithm. Plain
gradient descent converges far too slowly on separable-ish data.

## Formulation

Maximise the log-likelihood, equivalently minimise its negative plus a small
ridge term:

$$
J(\boldsymbol\beta) = -\sum_{i=0}^{n-1}\Bigl[y_i\log p_i + (1-y_i)\log(1-p_i)\Bigr] + \frac{\lambda}{2}\lVert\boldsymbol\beta\rVert^2,
\qquad p_i = \sigma(\mathbf x_i^{\mathsf T}\boldsymbol\beta) = \frac{1}{1 + e^{-\mathbf x_i^{\mathsf T}\boldsymbol\beta}}
$$

| Symbol | Meaning |
|---|---|
| $n,\ f$ | number of samples and features |
| $\mathbf x_i$ | row $i$ of $X$ (features of sample $i$) |
| $y_i$ | label of sample $i$, 0 or 1 |
| $\boldsymbol\beta$ | coefficients (output, length $f$) |
| $\sigma$ | logistic sigmoid |
| $p_i$ | predicted probability that $y_i = 1$ |
| $\lambda$ | L2 regularisation, $10^{-6}$ |
| $J$ | objective (negative log-likelihood + ridge) |

### Newton / IRLS step

$$
\mathbf g = X^{\mathsf T}(\mathbf p - \mathbf y) + \lambda\boldsymbol\beta, \qquad
H = X^{\mathsf T} W X + \lambda I, \qquad W = \operatorname{diag}\bigl(\max(p_i(1-p_i),\ 10^{-8})\bigr), \qquad
\boldsymbol\beta \leftarrow \boldsymbol\beta - H^{-1}\mathbf g
$$

Stop when $\lVert H^{-1}\mathbf g\rVert_2 < 10^{-8}$ (at most 1000 iterations).

| Symbol | Meaning |
|---|---|
| $\mathbf g$ | gradient of $J$ |
| $H$ | Hessian of $J$ (symmetric positive definite thanks to $\lambda I$ and the clamp) |
| $W$ | diagonal weights $p_i(1-p_i)$, clamped away from 0 |
| $I$ | $f \times f$ identity |
| $H^{-1}\mathbf g$ | Newton step, computed by a Cholesky solve (never an explicit inverse) |

Newton converges **quadratically** near the optimum, so typically 5–15
iterations suffice.

## Approach

Per iteration, four kernels, all in float64:

1. **`sampleTerms`** (warp per sample): $z_i = \mathbf x_i^{\mathsf T}\boldsymbol\beta$
   via a shuffle reduction. Lane 0 writes $W_i$ and the residual
   $r_i = p_i - y_i$.
2. **`weightedGram`**: $H = X^{\mathsf T}WX + \lambda I$, the tiled Gram
   kernel from [OLS](../033-ordinary-least-squares/), with the $A$-side tile
   pre-multiplied by $W_s$.
3. **`gradient`**: $g_j = \sum_s X_{sj} r_s + \lambda\beta_j$, one thread per
   feature.
4. **`newtonStep`** (one block): in-place Cholesky of $H$, forward and
   backward solves for $\boldsymbol\delta$, $\boldsymbol\beta \mathrel{-}= \boldsymbol\delta$,
   and a block reduction of $\lVert\boldsymbol\delta\rVert^2$.

Only the single scalar $\lVert\boldsymbol\delta\rVert^2$ is copied to the host
each iteration to decide whether to stop. $\boldsymbol\beta$ stays on the
device and is converted to float32 at the end.

## Cost analysis

$$
\text{per iteration:}\quad W \approx \underbrace{2nf}_{z} + \underbrace{2nf^2}_{H} + \underbrace{2nf}_{\mathbf g} + \underbrace{f^3/3}_{\text{Cholesky}}, \qquad \text{total} \approx T_{\text{it}}\cdot W
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs per Newton iteration |
| $T_{\text{it}}$ | number of iterations (≈ 10) |

The benchmark ($16 \times 8$) is entirely latency-bound: about 10 iterations ×
(4 launches + 1 small device-to-host copy). A persistent single-block kernel
running the whole loop on the device would remove the host round trips.

## Pitfalls

- **Stopping rule.** The reference stops on the *step* norm, not the
  gradient norm. Using the same criterion makes the result agree to far below
  the tolerance.
- **Separable data.** Without $\lambda$ and the clamp on $W$, $H$ can become
  singular as $p_i \to 0/1$.
- **Float32.** It would break convergence at the $10^{-8}$ step threshold.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-2`,
including nearly separable data.

## Related

- [Ordinary Least Squares](../033-ordinary-least-squares/), [Categorical Cross-Entropy](../025-categorical-cross-entropy-loss/).

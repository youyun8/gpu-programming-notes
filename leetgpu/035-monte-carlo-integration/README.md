---
title: Monte Carlo Integration
platform: LeetGPU
upstream: medium/35_monte_carlo_integration
url: https://leetgpu.com/challenges/monte-carlo-integration
difficulty: medium
tags: [reduction, statistics, two-pass]
status: solved
---

# Monte Carlo Integration

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/monte-carlo-integration)

## Problem

Estimate $\int_a^b f(x)\,dx$ from $n$ precomputed samples $y_i = f(x_i)$,
with $x_i$ uniform on $[a, b]$ ($1 \le n \le 10^8$, $-1000 \le a < b \le 1000$,
$\lvert y_i \rvert \le 10^4$; benchmark $n = 10^7$; tolerance `1e-2`). The
random sampling is already done, so the GPU work is a mean. The page also
covers why the estimator works and how accurate it is.

## Formulation

$$
I = \int_a^b f(x)\,dx \;\approx\; \hat I_n = (b - a)\cdot\frac{1}{n}\sum_{i=0}^{n-1} y_i, \qquad y_i = f(x_i),\ x_i \sim \mathcal U[a, b]
$$

| Symbol | Meaning |
|---|---|
| $a,\ b$ | integration limits |
| $f$ | integrand (only its sampled values are given) |
| $x_i$ | independent uniform sample points on $[a, b]$ |
| $y_i$ | function values (float32 input `y_samples`) |
| $n$ | number of samples |
| $I$ | exact integral |
| $\hat I_n$ | Monte Carlo estimate, written to `result[0]` |

Because $\mathbb E[f(x)] = I/(b-a)$ for uniform $x$, the estimator is unbiased,
and by the central limit theorem its error shrinks like $1/\sqrt n$:

$$
\mathbb E[\hat I_n] = I, \qquad \operatorname{Std}[\hat I_n] = \frac{(b-a)\,\sigma_f}{\sqrt n}, \qquad \sigma_f^2 = \operatorname{Var}_{x\sim\mathcal U[a,b]}[f(x)]
$$

| Symbol | Meaning |
|---|---|
| $\mathbb E,\ \operatorname{Std},\ \operatorname{Var}$ | expectation, standard deviation, variance over the random samples |
| $\sigma_f$ | standard deviation of $f$ under the uniform distribution |

The GPU's job is only to compute the **sample mean** accurately. Its own
rounding error must stay far below the statistical error, which is easy with
float64 partial sums.

## Approach

The two-pass reduction from [Reduction](../004-reduction/):

1. **`partialSums`**: grid-stride `float4` loads, float32 per-thread sums,
   and a float64 block reduction into one partial per block (≤ 1024 blocks).
2. **`finalize`**: one block adds the partials in float64 and writes
   $(b - a)\cdot\text{sum}/n$, computing $b - a$ in double.

## Cost analysis

$$
Q = 4n \ \text{bytes}, \qquad W = n, \qquad T_{\min} = \frac{4n}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes (read every sample once) |
| $W$ | additions |
| $\beta$ | DRAM bandwidth |

Benchmark: 40 MB, so ≈ 20 µs at 2 TB/s.

## Pitfalls

- **$(b-a)$ in float32.** With $a, b$ up to $\pm1000$ this is exact enough,
  but computing it in double costs nothing.
- **Dividing early.** Dividing each sample by $n$ before summing wastes
  precision on tiny numbers. Divide once at the end.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-2`.

## Related

- [Reduction](../004-reduction/), [Mean Squared Error](../027-mean-squared-error/).

---
title: Sigmoid Linear Unit
platform: LeetGPU
upstream: easy/52_silu
url: https://leetgpu.com/challenges/sigmoid-linear-unit
difficulty: easy
tags: [elementwise, activation, transcendental]
status: solved
---

# Sigmoid Linear Unit

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/sigmoid-linear-unit)

## Problem

Apply SiLU (also called *Swish-1*) elementwise to $N$ float32 values
($N \le 10^4$ in the tests, $5\times10^4$ in the benchmark; inputs in
$[-100, 100]$; tolerance `1e-5`). SiLU is the activation in the SwiGLU MLPs
of LLaMA and most modern LLMs.

## Formulation

$$
\operatorname{SiLU}(x) = x\,\sigma(x) = \frac{x}{1 + e^{-x}}, \qquad \sigma(x) = \frac{1}{1 + e^{-x}}
$$

| Symbol | Meaning |
|---|---|
| $x$ | input value |
| $\sigma$ | logistic sigmoid |
| $\operatorname{SiLU}(x)$ | output value |

Limits: $\operatorname{SiLU}(x) \to x$ as $x \to +\infty$ and $\to 0^-$ as
$x \to -\infty$, with a minimum of about $-0.278$ at $x \approx -1.278$.

## Approach

One thread per element computing `x / (1.0f + expf(-x))`, which is one
exponential and one division, with no branches.

**Behaviour at the extremes.** For $x = -100$, $e^{100}$ overflows to
$+\infty$, and $x/\infty = -0$. That is the correct limit and matches
PyTorch. For $x = +100$, $e^{-100}$ underflows to 0, giving $x/1 = x$. No
special-casing is needed. The alternative form `x * (1/(1+e^{-x}))` behaves
the same way.

## Cost analysis

$$
Q = 8N\ \text{bytes}, \qquad W \approx N\,(c_{\exp} + c_{\div})
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes (read and write each element) |
| $W$ | instruction work; $c_{\exp} \approx 10$–$20$ instructions for accurate `expf`, $c_{\div} \approx 10$ for IEEE division |

At $N = 5\times10^4$ (200 KB) this is a single-wave kernel dominated by launch
latency. At large $N$ it would be memory-bound on most GPUs, but closer to the
balance point than ReLU because of the transcendental.

## Pitfalls

- **`__expf` / `__fdividef`** are faster intrinsics with larger error.
  `__fdividef` also returns 0 instead of $-0$ for huge denominators, which is
  harmless. The accurate versions keep far inside `1e-5`.
- **Computing $\sigma$ as `expf(x)/(1+expf(x))`** overflows to NaN
  ($\infty/\infty$) for large positive $x$.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-5`,
including $\pm100$.

## Related

- [SwiGLU](../054-swiglu/), [SwiGLU MLP Block](../084-swiglu-mlp-block/), [Sigmoid](../068-sigmoid/).
- Tensara [Swish](../../tensara/swish/).

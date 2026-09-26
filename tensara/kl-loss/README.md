---
title: Kullback-Leibler Divergence
platform: Tensara
upstream: kl-loss
url: https://tensara.org/problems/kl-loss
difficulty: medium
tags: [loss, elementwise, numerics]
status: solved
---

# Kullback-Leibler Divergence

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/kl-loss)

## Problem

Element-wise Kullback–Leibler divergence contributions between a target
distribution $P$ and a predicted distribution $Q$, each a float32 vector of
length $N$. The reference clamps both inputs at $\epsilon = 10^{-10}$ before
taking logs and zeroes the entries where the target is not positive. The
check is tight: `rtol = atol = 1e-5`.

## Formulation

$$
D_{\mathrm{KL}}(P\,\Vert\,Q) = \sum_i p_i \log\frac{p_i}{q_i}
$$

| Symbol | Meaning |
|---|---|
| $P, Q$ | target and predicted distributions |
| $p_i, q_i$ | their probabilities (`targets[i]`, `predictions[i]`) |
| $D_{\mathrm{KL}}$ | KL divergence (non-negative, 0 iff $P = Q$) |

The required output is the element-wise term, exactly as the reference
computes it:

$$
\tilde{p}_i = \max(p_i, \epsilon), \quad \tilde{q}_i = \max(q_i, \epsilon), \qquad
\text{out}_i = \begin{cases} \tilde{p}_i\bigl(\ln\tilde{p}_i - \ln\tilde{q}_i\bigr), & p_i > 0 \\ 0, & p_i \le 0 \end{cases}
$$

| Symbol | Meaning |
|---|---|
| $\epsilon$ | clamp value $10^{-10}$ |
| $\tilde{p}_i, \tilde{q}_i$ | clamped probabilities |
| $\ln$ | natural logarithm (`logf`) |
| $\text{out}_i$ | element-wise contribution; may be negative |

The zero case follows the limit $\lim_{p\to0^+} p\ln p = 0$.

## Approach

A grid-stride elementwise map: two loads, two `logf`, one subtraction, one
multiply, one select, one store. Computing $\ln\tilde{p} - \ln\tilde{q}$ as
two logarithms (as the reference does) rather than $\ln(\tilde{p}/\tilde{q})$
keeps the rounding identical to PyTorch's.

## Cost analysis

$$
Q = 12N\ \text{bytes}, \qquad T_{\min} = \frac{12N}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes: two inputs, one output |
| $\beta$ | DRAM bandwidth |

Two `logf` per element are about 40 instructions, still hidden behind the
12 bytes of memory traffic.

## Pitfalls

- **Reproduce the clamping**: $\ln(0) = -\infty$ and $0\cdot(-\infty) =$ NaN.
- **Condition on the unclamped target** ($p_i > 0$), as the reference's
  `torch.where(targets > 0, ...)` does.
- **`__logf`** (fast math) has absolute error around $2^{-21.4}$, which can
  exceed `atol = 1e-5` after the multiply for larger $p$; use `logf`.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Huber Loss](../huber-loss/), [Log Softmax](../log-softmax/),
  LeetGPU [Categorical Cross-Entropy](../../leetgpu/025-categorical-cross-entropy-loss/).

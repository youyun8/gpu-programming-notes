---
title: DPO Sequence Loss
platform: LeetGPU
upstream: medium/108_dpo_sequence_loss
url: https://leetgpu.com/challenges/dpo-sequence-loss
difficulty: medium
tags: [reduction, rl, loss, rlhf, numerical-stability]
status: solved
---

# DPO Sequence Loss

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/dpo-sequence-loss)

## Problem

The **Direct Preference Optimisation** loss. Each of $B$ preference pairs
has four summed sequence log-probabilities: the chosen and rejected responses,
each under the trainable policy and under the frozen reference model. Return
the mean loss for temperature $\beta$ (tolerance `1e-4`). DPO fine-tunes LLMs
on human preferences without training a reward model or running RL.

## Formulation

$$
z_i = \beta\Bigl[\bigl(\ell^+_i - \ell^-_i\bigr) - \bigl(\ell^{+,\text{ref}}_i - \ell^{-,\text{ref}}_i\bigr)\Bigr], \qquad
\mathcal L = \frac1B\sum_{i=0}^{B-1} -\log\sigma(z_i) = \frac1B\sum_i \operatorname{softplus}(-z_i)
$$

$$
\operatorname{softplus}(x) = \log(1 + e^{x}) = \max(x, 0) + \log\bigl(1 + e^{-\lvert x\rvert}\bigr)
$$

| Symbol | Meaning |
|---|---|
| $B$ | number of preference pairs |
| $\ell^+_i,\ \ell^-_i$ | policy log-probability of the chosen / rejected response $\log\pi_\theta(y^\pm\mid x)$ |
| $\ell^{\pm,\text{ref}}_i$ | same under the frozen reference policy |
| $\beta$ | temperature: how strongly to deviate from the reference |
| $z_i$ | implicit reward margin: how much more the policy prefers $y^+$ than the reference does |
| $\sigma$ | logistic sigmoid |
| softplus | $\log(1+e^x)$; the right-hand form is overflow-free |
| $\mathcal L$ | mean DPO loss |

**Why the stable form.** $\log(1 + e^{x})$ computed literally overflows for
$x > 88$ (giving $\infty$) and loses all precision for $x < -17$
($1 + e^{x}$ rounds to 1). The rewritten form evaluates $e^{-\lvert x\rvert} \le 1$,
and `log1pf` keeps full relative precision for small arguments.

## Approach

$B$ is small (one value per pair), so a **single block of 1024 threads**
suffices: grid-stride over pairs, compute $z$ and the stable softplus in
float32, accumulate in float64, then a warp-shuffle and shared-memory
reduction. Thread 0 writes $\text{sum}/B$.

## Cost Analysis

$$
Q = 16B\ \text{bytes}, \qquad W \approx B\,(c_{\exp} + c_{\log} + 8)
$$

| Symbol | Meaning |
|---|---|
| $Q$ | bytes: four float arrays read once |
| $W$ | per pair: one `expf`, one `log1pf`, a few adds and multiplies |

Microseconds at any realistic $B$. The expensive part of DPO, computing the
four sequence log-probabilities, happens in the model's forward passes.

## Pitfalls

- **Naive `-log(sigmoid(z))`**: `sigmoid(-100)` underflows to 0, giving
  $-\log 0 = \infty$.
- **Sign.** The loss is `softplus(-z)`, which is small when the policy prefers
  the chosen response more strongly than the reference does.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-4`,
including margins of $\pm100$.

## Related

- [PPO Clipped Loss](../107-ppo-clipped-surrogate-loss/), [GRPO Loss](../109-grpo-surrogate-loss/),
  [Categorical Cross-Entropy](../025-categorical-cross-entropy-loss/). Tensara [Softplus](../../tensara/soft-plus/).

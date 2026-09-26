---
title: GRPO Surrogate Loss
platform: LeetGPU
upstream: medium/109_grpo_surrogate_loss
url: https://leetgpu.com/challenges/grpo-surrogate-loss
difficulty: medium
tags: [reduction, rl, loss, rlhf]
status: solved
---

# GRPO Surrogate Loss

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/grpo-surrogate-loss)

## Problem

The **GRPO** (Group Relative Policy Optimisation, DeepSeekMath/DeepSeek-R1)
objective. For each of $B$ prompts, $G$ responses were sampled and scored.
Each response's advantage is its reward standardised **within its group**,
so no critic network is needed. That advantage is applied to every one of the
response's $S$ tokens in a PPO-style clipped objective, plus a KL penalty to
a reference policy. Inputs: rewards $(B, G)$ and three log-probability
tensors $(B, G, S)$. Output: the scalar loss (tolerance `1e-4`).

## Formulation

$$
\mu_b = \frac1G\sum_g R_{b,g}, \qquad \sigma_b = \sqrt{\frac1G\sum_g (R_{b,g} - \mu_b)^2}, \qquad
A_{b,g} = \frac{R_{b,g} - \mu_b}{\sigma_b + 10^{-8}}
$$

$$
r_{b,g,s} = e^{\log\pi - \log\pi^{\text{old}}}, \qquad
d_{b,g,s} = \log\pi^{\text{ref}} - \log\pi, \qquad
K_{b,g,s} = e^{d} - d - 1
$$

$$
\mathcal L = -\frac{1}{BGS}\sum_{b,g,s}\Bigl[\min\bigl(rA_{b,g},\ \operatorname{clip}(r, 1\pm\varepsilon)A_{b,g}\bigr) - \beta K_{b,g,s}\Bigr]
$$

| Symbol | Meaning |
|---|---|
| $B,\ G,\ S$ | prompts, responses per prompt (group size), tokens per response |
| $R_{b,g}$ | scalar reward of response $g$ to prompt $b$ |
| $\mu_b,\ \sigma_b$ | group mean and **population** standard deviation |
| $A_{b,g}$ | group-relative advantage, shared by all $S$ tokens of the response |
| $\log\pi,\ \log\pi^{\text{old}},\ \log\pi^{\text{ref}}$ | log-probabilities of token $(b, g, s)$ under the current, the sampling, and the reference policy |
| $r$ | importance ratio |
| $\varepsilon$ | clip range (`clip_eps`) |
| $d$ | log-ratio of reference to current policy |
| $K$ | the "$k_3$" KL estimator $e^d - d - 1 \ge 0$ (unbiased for $\mathrm{KL}(\pi\,\Vert\,\pi^{\text{ref}})$ in expectation, and always non-negative) |
| $\beta$ | KL penalty weight |
| $\mathcal L$ | loss (negative objective) |

## Approach

1. **`groupAdvantages`**: one thread per prompt ($G$ is small, e.g. 8–64).
   It computes the float64 mean and population variance, then writes the $G$
   advantages to a scratch buffer.
2. **`tokenSums`**: grid-stride over all $BGS$ tokens (64-bit index). The
   advantage is `adv[i / S]` (response index = token index / $S$). Compute the
   ratio, clip, min, and the $k_3$ term, accumulate in float32 per thread,
   then a float64 block reduction into partials.
3. **`finalize`**: $-\text{sum}/(BGS)$.

## Cost analysis

$$
Q \approx 12BGS + 8BG\ \text{bytes}, \qquad W \approx BGS\,(2c_{\exp} + 10)
$$

| Symbol | Meaning |
|---|---|
| $Q$ | bytes: three token-level arrays, plus rewards and advantages |
| $W$ | per token: two `expf` plus about 10 FLOPs |

The kernel is memory-bound. The advantage read `adv[i / S]` is the same value
for $S$ consecutive tokens, so it is always a cache hit.

## Pitfalls

- **Population std** (`unbiased=False`), plus $10^{-8}$ added to $\sigma$,
  not to $\sigma^2$.
- **KL estimator.** It is $k_3 = e^{d} - d - 1$ with $d = \log\pi^{\text{ref}} - \log\pi$.
  The sign of $d$ matters.
- **Constant groups** ($\sigma = 0$) give $A = 0/10^{-8} = 0$, which is
  correct, not NaN.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-4`,
including $G = 1$ (all advantages 0) and constant-reward groups.

## Related

- [PPO Clipped Loss](../107-ppo-clipped-surrogate-loss/), [DPO Loss](../108-dpo-sequence-loss/).

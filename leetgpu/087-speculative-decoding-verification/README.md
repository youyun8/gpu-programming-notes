---
title: Speculative Decoding Verification
platform: LeetGPU
upstream: medium/87_speculative_decoding_verification
url: https://leetgpu.com/challenges/speculative-decoding-verification
difficulty: medium
tags: [sampling, llm, speculative-decoding, scan]
status: solved
---

# Speculative Decoding Verification

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/speculative-decoding-verification)

## Problem

The acceptance step of **speculative decoding**. A small draft model
proposed $T$ tokens per sequence; the large target model scored all of them
in one pass. For each of $B$ sequences, accept draft tokens left to right
with the standard rejection rule. At the first rejection, resample from the
residual distribution and stop. If all $T$ are accepted, draw a bonus token
from the target distribution. Output $(B, T+1)$ token ids, zero-padded.
Given the same uniform samples, the result must match exactly.

## Formulation

At draft position $i$ with draft token $t_i$:

$$
\alpha_i = \min\!\Bigl(1,\ \frac{q_i(t_i)}{p_i(t_i)}\Bigr), \qquad \text{accept } t_i \iff u_i < \alpha_i
$$

On the first rejection at position $i$, sample from the **residual distribution**

$$
r_i(v) = \frac{\max\bigl(0,\ q_i(v) - p_i(v)\bigr)}{\sum_{v'}\max\bigl(0,\ q_i(v') - p_i(v')\bigr)}\quad (\text{uniform } 1/V \text{ if the sum is } 0)
$$

and if all $T$ tokens are accepted, sample a bonus token from $q_{T-1}$. Sampling uses the inverse CDF:

$$
\operatorname{sample}(\pi, u) = \min\Bigl\{\, v : \sum_{v' \le v} \pi(v') \ge u \,\Bigr\}\ \ (\text{clamped to } V-1)
$$

| Symbol | Meaning |
|---|---|
| $B,\ T,\ V$ | batch size, number of draft tokens, vocabulary size |
| $t_i$ | draft token at position $i$ |
| $p_i(v)$ | draft model probability of token $v$ at position $i$ |
| $q_i(v)$ | target model probability |
| $u_i$ | uniform sample for the acceptance test at position $i$ |
| $\alpha_i$ | acceptance probability |
| $r_i$ | residual distribution used after a rejection |
| $u_T$ | the extra uniform sample (index $T$) used for resampling or the bonus token |
| sample$(\pi, u)$ | inverse-CDF draw (`torch.searchsorted(cumsum(π), u)`) |

**Why it is exact.** Accepting with probability
$\min(1, q/p)$ and otherwise sampling from $r$ yields tokens distributed
exactly according to the target $q$ (Leviathan et al., Chen et al. 2023). The
large model's output distribution is preserved while it verifies $T$ tokens
per forward pass.

## Approach

**One block of 1024 threads per sequence:**

1. Zero the output row.
2. Walk $i = 0..T-1$ sequentially (the chain stops at the first rejection):
   - the acceptance test is a scalar computation (every thread evaluates it
     identically, so control flow stays uniform);
   - accepted → thread 0 writes $t_i$;
   - rejected → a **block reduction** computes $\sum\max(0, q - p)$ over $V$;
     then `inverseCdf` runs a chunked **block scan** of the weights
     (warp `__shfl_up_sync` scan, warp totals, and a carry across chunks of
     1024 vocab entries), finding the first $v$ with running sum $\ge u_T$
     via `atomicMin`, with early exit. Write it and `return`.
3. If all $T$ are accepted, run the same inverse CDF over $q_{T-1}$ for the
   bonus token.

The weight function is passed to `inverseCdf` as a lambda, so the same scan
code serves the residual, uniform and target distributions.

## Cost analysis

$$
W \le B\,\bigl(T + 3V\bigr), \qquad Q \le 4B\,(T + 2V\cdot 2)
$$

| Symbol | Meaning |
|---|---|
| $W$ | work: $T$ scalar tests plus at most one residual reduction and one scan over $V$ per sequence |
| $Q$ | bytes: at most two full vocabulary rows of $p$ and $q$ read per sequence (the tensors are $B\times T\times V$, but only the rows actually visited are read) |

With $V$ around $3\times10^4$–$1.5\times10^5$ in real LLMs, the scan
dominates. A single block per sequence keeps it simple and synchronisation-free.

## Pitfalls

- **Which uniform sample.** The acceptance tests use $u_0..u_{T-1}$. The
  resample and bonus both use $u_T$, as in the reference.
- **Division by $p(t_i) = 0$** gives $\infty$, so $\alpha = 1$ (always
  accept), which matches Python's float semantics.
- **`searchsorted` semantics**: the first index with cumsum **$\ge$** $u$
  (left insertion point), clamped to $V-1$ against rounding at the end of the
  CDF.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) with exact
token equality, including all-accepted (bonus path), immediate rejection,
and zero-residual (uniform fallback) cases.

## Related

- [Top-p Sampling](../060-top-p-sampling/), [Prefix Sum](../016-prefix-sum/),
  [Adder Transformer](../076-adder-transformer/) (autoregressive decoding).

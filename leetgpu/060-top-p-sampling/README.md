---
title: Top-P Sampling
platform: LeetGPU
upstream: medium/60_top_p_sampling
url: https://leetgpu.com/challenges/top-p-sampling
difficulty: medium
tags: [sampling, softmax, selection, llm, bit-tricks]
status: solved
---

# Top-P Sampling

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/top-p-sampling)

## Problem

Nucleus (top-$p$) sampling of one token from a vocabulary of $V$ logits
($3 \le V \le 5\times10^4$, logits in $[-100, 100]$, $0 < p \le 1$; benchmark
$V = 5\times10^4$). Convert to probabilities, keep the smallest set of most
likely tokens whose mass reaches $p$, renormalise, and sample with the given
seed. It is the default decoding strategy of most LLM APIs. The textbook
implementation sorts the whole vocabulary. This solution **finds the nucleus
without sorting**.

## Formulation

$$
\pi_t = \frac{e^{z_t - m}}{\sum_{u} e^{z_u - m}}, \qquad
\pi_{(0)} \ge \pi_{(1)} \ge \dots, \qquad
c = \min\Bigl\{ c' : \sum_{r=0}^{c'} \pi_{(r)} \ge p \Bigr\}, \qquad
\mathcal N = \{(0), \dots, (c)\}
$$

$$
\Pr[\text{sample} = t] = \frac{\pi_t}{\sum_{u\in\mathcal N}\pi_u}\ \ \text{for } t\in\mathcal N, \qquad 0 \text{ otherwise}
$$

| Symbol | Meaning |
|---|---|
| $V$ | vocabulary size |
| $z_t$ | logit of token $t$ |
| $m$ | $\max_t z_t$ (softmax stabiliser) |
| $\pi_t$ | softmax probability of token $t$ |
| $\pi_{(r)}$ | $r$-th largest probability (descending order statistics) |
| $c$ | cutoff rank: `searchsorted(cumsum, p)` in the reference |
| $\mathcal N$ | the nucleus: the top $c + 1$ tokens |
| $p$ | nucleus mass threshold |

### Nucleus as a Threshold

Because the nucleus is a *top set*, it equals $\{t : \pi_t \ge T\}$ for the
threshold $T = \pi_{(c)}$, which is the largest value with enough mass above
it:

$$
T = \max\Bigl\{ \tau : \sum_{t\,:\,\pi_t \ge \tau} \pi_t \ \ge\ p \Bigr\}
$$

| Symbol | Meaning |
|---|---|
| $T$ | probability of the least likely nucleus token |
| $\tau$ | candidate threshold |

**Positive floats compare like their bit patterns** (the sign bit is 0 and
the exponent sits above the mantissa). $T$ can therefore be found with a
**bitwise binary search** over the 32-bit pattern, from the MSB down: tentatively
set bit $b$, measure the mass of $\{\pi_t \ge \text{candidate}\}$ with a block
reduction, and keep the bit if the mass is still $\ge p$. After 32 steps the
pattern is exactly $T$.

### Sampling by Inverse CDF

Draw $u \in [0, 1)$ and return the first nucleus token (in index order) at
which the running nucleus mass exceeds $u \cdot \sum_{\mathcal N}\pi$. Any
fixed order of the nucleus gives the right distribution.

## Approach

Everything runs in **one block of 1024 threads**; $V \le 50\,000$ is ~50
elements per thread:

1. **Softmax statistics.** A block max of $z$, then a block sum of
   $e^{z - m}$, gives $1/\sum$.
2. **Threshold search.** 32 iterations, each a grid-stride pass recomputing
   $\pi_t$ (cheaper than storing 50k floats) plus a block sum.
3. **Nucleus mass.** $\sum_{\pi_t \ge T}\pi_t$.
4. **Random number.** SplitMix64 of the seed gives a 64-bit hash, whose top
   24 bits form a float in $[0, 1)$. It is scaled by the nucleus mass.
5. **Inverse CDF** over index order, 1024 tokens per chunk. A block
   inclusive scan (warp shuffles plus warp totals, plus a carry) finds the
   first index whose running mass exceeds the target, using `atomicMin` on a
   shared "hit" slot and stopping early. A fallback returns the last nucleus
   token if rounding leaves the target just above the total.

## Cost Analysis

$$
W \approx V\,(c_{\exp} + 1)\cdot(2 + 32 + 1 + 1), \qquad Q = 4V \ \text{bytes (L1/L2-resident after the first pass)}
$$

| Symbol | Meaning |
|---|---|
| $W$ | work: each of ~36 passes recomputes one `expf` per token |
| $c_{\exp}$ | cost of one accurate `expf` |
| $Q$ | the logits are read from DRAM once (200 KB), then hit in cache |

About $1.8$M exponentials in total, tens of microseconds on one SM. A full
sort of 50k keys would take longer and need several kernels.

## Pitfalls

- **Exact token vs. reference.** The reference samples with
  `torch.multinomial` after seeding PyTorch's generator (Philox on GPU,
  mt19937 on CPU). That stream cannot be reproduced from CUDA C++, so a
  correct sampler only returns the *same token* when $\lvert\mathcal N\rvert = 1$.
  The local runner therefore checks that the token is **in the nucleus**,
  which is the property a correct sampler guarantees. The platform's own
  grader may differ.
- **Ties at $T$.** All tokens with $\pi_t = T$ are included. Sorting-based
  implementations may include only some of them. The difference has
  measure zero for continuous inputs.
- **`>=` vs. `>`.** `searchsorted(right=False) + 1` corresponds to "smallest
  prefix with cumsum ≥ p", which is the ≥ in the threshold definition.

## Verification

Nucleus membership and exact $\lvert\mathcal N\rvert = 1$ cases are checked
on all LeetGPU cases in [cuemu](../../tools/cuemu/README.md). The nucleus
itself (the set) was compared with a sort-based Python implementation for
thousands of random vocabularies.

## Related

- [Top-K Selection](../029-top-k-selection/) (radix select on bit patterns), [Softmax](../005-softmax/),
  [Speculative Decoding Verification](../087-speculative-decoding-verification/).

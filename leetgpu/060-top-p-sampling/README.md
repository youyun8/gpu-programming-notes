---
title: Top-p Sampling
platform: LeetGPU
upstream: medium/60_top_p_sampling
url: https://leetgpu.com/challenges/top-p-sampling
difficulty: medium
tags: [sampling, softmax, selection, llm]
status: solved
---

# Top-p Sampling

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/top-p-sampling)

## Problem
Nucleus sampling: softmax, then take the smallest top set with mass ≥ `p`,
renormalize, and sample using a seed.

## Approach
A single block (vocab ≤ 50k), **without sorting**:
1. Compute the softmax max and denominator with block reductions.
2. Positive float probabilities are order-preserving as `uint32` bit
   patterns. A 32-step bitwise search finds the largest threshold `T` with
   `mass(prob ≥ T) ≥ p`. `{prob ≥ T}` is exactly the prefix of the
   descending sort that `searchsorted(cumsum, p) + 1` selects.
3. Draw `u ∈ [0,1)` from the seed (SplitMix64) and walk the nucleus with a
   block scan until the cumulative mass passes `u · mass`.

## Pitfalls
- **The reference draws with `torch.multinomial`.** Its RNG stream (Philox
  on GPU, mt19937 on CPU) can't be reproduced from CUDA C++. The sampled
  distribution is correct, but the exact token only matches the reference when
  the nucleus has one element. The local test runner therefore checks
  nucleus membership for this challenge.

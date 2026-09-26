---
title: Speculative Decoding Verification
platform: LeetGPU
upstream: medium/87_speculative_decoding_verification
url: https://leetgpu.com/challenges/speculative-decoding-verification
difficulty: medium
tags: [sampling, llm, scan]
status: solved
---

# Speculative Decoding Verification

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/speculative-decoding-verification)

## Problem
Accept or reject draft tokens (`u < min(1, q/p)`), resample from
`max(0, q-p)` on the first rejection, or sample a bonus token if all are accepted.

## Approach
One block per sequence. The walk over draft positions is sequential and stops
at the first rejection. The vocabulary-sized work (building the residual
distribution, normalizing it, and the inverse-CDF search, i.e.
`searchsorted` = the first index where the running sum ≥ `u`) is spread over
the block with block scans.

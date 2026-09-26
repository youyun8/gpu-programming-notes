---
title: Attention with Linear Biases
platform: LeetGPU
upstream: medium/55_attn_w_linear_bias
url: https://leetgpu.com/challenges/attention-with-linear-biases
difficulty: medium
tags: [attention, alibi, gemm, softmax]
status: solved
---

# Attention with Linear Biases

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/attention-with-linear-biases)

## Problem
`softmax(QKᵀ/√d + α(i−j))·V` with `M, N ≤ 2048`, `d ≤ 1024`.

## Approach
`d` up to 1024 is too large to keep K/V tiles resident per query warp, so this
uses three passes over an `M×N` score buffer (≤16 MB):
1. An "NT" register-blocked SGEMM computes `QKᵀ`, with the scale and the ALiBi
   bias `α(i−j)` **fused into the epilogue**.
2. A row softmax, one warp per row.
3. An "NN" SGEMM multiplies the probabilities by `V`.

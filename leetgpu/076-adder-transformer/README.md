---
title: Adder Transformer Inference
platform: LeetGPU
upstream: medium/76_adder_transformer
url: https://leetgpu.com/challenges/adder-transformer-inference
difficulty: medium
tags: [transformer, inference, decoding]
status: solved
---

# Adder Transformer Inference

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/adder-transformer-inference)

## Problem
Greedy decoding (11 steps) of a 10-parameter, 1-layer, 2-dim transformer that
adds two 10-digit numbers; output the logits of each step.

## Approach
Only the **last** position's logits are needed at each step. With a single
layer, every position's K and V depend only on that position's token and
index, so K/V are computed once per token and cached. Each decode step is
then `O(seq_len)`, and one thread can run a whole sequence's decode loop. The
fixed model constants (ω, attention scale) are computed on the host in double,
exactly as the reference does.

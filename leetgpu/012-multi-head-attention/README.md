---
title: Multi-Head Attention
platform: LeetGPU
upstream: hard/12_multi_head_attention
url: https://leetgpu.com/challenges/multi-head-attention
difficulty: hard
tags: [attention, flash-attention, multi-head]
status: solved
---

# Multi-Head Attention

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/multi-head-attention)

## Problem
Multi-head self-attention on `N × d_model` inputs with `h` heads (`d_k` can be as large as 1024).

## Approach
Head `i` is just columns `[i·d_k, (i+1)·d_k)` of Q/K/V and of the output, so
"split heads" and "concat" are **strides** and need no data movement. One
fused flash-attention kernel runs per (4-row block, head): lane-per-key
scoring, online softmax, register accumulators. Because `d_k` can reach 1024,
K and V tiles are streamed through shared memory in **128-wide slices of the
head dimension**. Shared memory stays bounded, and the accumulator loop over
slices is unrolled so it stays in registers.

---
title: Attention with Sinks
platform: LeetGPU
upstream: medium/112_attention_with_sinks
url: https://leetgpu.com/challenges/attention-with-sinks
difficulty: medium
tags: [attention, sliding-window, streaming-llm]
status: solved
---

# Attention with Sinks

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/attention-with-sinks)

## Problem
Each query attends to the first `num_sinks` keys plus a causal sliding window.

## Approach
Flash kernel that only visits the key tiles that can be allowed for its 8
rows, `[0, sinks)` and `[first_row − w + 1, last_row]`, and masks per lane.
Work is `O(M·(sinks + w))` instead of `O(M²)`.

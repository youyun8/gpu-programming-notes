---
title: Rotary Positional Embedding
platform: LeetGPU
upstream: medium/61_rope_embedding
url: https://leetgpu.com/challenges/rotary-positional-embedding
difficulty: medium
tags: [elementwise, rope, llm]
status: solved
---

# Rotary Positional Embedding

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/rotary-positional-embedding)

## Problem
Rotary position embedding: `q·cos + rotate_half(q)·sin`.

## Approach
One thread per element pair `(j, j + D/2)`. Both outputs need the same two
inputs, so every input element is read exactly once:
`out[j] = x1·cos − x2·sin`, `out[j+D/2] = x2·cos + x1·sin`.

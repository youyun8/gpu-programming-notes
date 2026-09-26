---
title: Interleave Arrays
platform: LeetGPU
upstream: easy/63_interleave
url: https://leetgpu.com/challenges/interleave-arrays
difficulty: easy
tags: [memory-bound, vectorized]
status: solved
---

# Interleave Arrays

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/interleave-arrays)

## Problem
`output = [A0, B0, A1, B1, …]`.

## Approach
Thread `i` reads `A[i]` and `B[i]` (coalesced) and writes them together as one
`float2` to `output[2i]` — an 8-byte store instead of two strided 4-byte stores.

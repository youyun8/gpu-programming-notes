---
title: Rainbow Table
platform: LeetGPU
upstream: easy/24_rainbow_table
url: https://leetgpu.com/challenges/rainbow-table
difficulty: easy
tags: [hashing, compute-bound, integer]
status: solved
---

# Rainbow Table

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/rainbow-table)

## Problem
Apply 32-bit FNV-1a hashing `R` times to every input integer.

## Approach
Embarrassingly parallel and compute-bound: each thread loads one value,
iterates the hash `R` times in registers, stores once. Unsigned 32-bit
arithmetic wraps exactly like the reference's `& 0xFFFFFFFF`.

## Pitfalls
- Use `unsigned int` — signed overflow is undefined behaviour in C++.

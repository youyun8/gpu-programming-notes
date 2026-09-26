---
title: Reverse Array
platform: LeetGPU
upstream: easy/19_reverse_array
url: https://leetgpu.com/challenges/reverse-array
difficulty: easy
tags: [in-place, memory-bound]
status: solved
---

# Reverse Array

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/reverse-array)

## Problem
Reverse a float32 array in place.

## Approach
Launch `N/2` threads; thread `i` swaps `data[i]` and `data[N-1-i]`. Each pair
is owned by exactly one thread, so there is no race even though the update
is in place. (The middle element of an odd-length array stays put.)

## Pitfalls
- Launching `N` threads that each write `data[N-1-i] = data[i]` races: half of
  them read values another thread already overwrote.

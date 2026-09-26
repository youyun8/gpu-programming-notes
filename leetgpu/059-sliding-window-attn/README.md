---
title: Sliding Window Self-Attention
platform: LeetGPU
upstream: hard/59_sliding_window_attn
url: https://leetgpu.com/challenges/sliding-window-self-attention
difficulty: hard
tags: [attention, sliding-window, flash-attention]
status: solved
---

# Sliding Window Self-Attention

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/sliding-window-self-attention)

## Problem
Each query attends to keys within `±window_size`.

## Approach
The flash kernel visits only the key range `[first_row − w, last_row + w]` of
its 8 rows and masks per lane, which is `O(M·w)` work.

---
title: Kullback-Leibler Divergence
platform: Tensara
upstream: kl-loss
url: https://tensara.org/problems/kl-loss
difficulty: medium
tags: [loss, elementwise]
status: solved
---

# Kullback-Leibler Divergence

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/kl-loss)

## Problem
Elementwise KL term `t·(log t − log p)` with clamping at 1e-10, zero where `t ≤ 0`.

## Approach
Elementwise kernel that reproduces the reference's clamping.

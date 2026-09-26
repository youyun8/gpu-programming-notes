---
title: Huber Loss
platform: Tensara
upstream: huber-loss
url: https://tensara.org/problems/huber-loss
difficulty: easy
tags: [loss, elementwise]
status: solved
---

# Huber Loss

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/huber-loss)

## Problem
Elementwise smooth-L1 (β = 1).

## Approach
`|d| < 1 ? d²/2 : |d| − 1/2`, elementwise.

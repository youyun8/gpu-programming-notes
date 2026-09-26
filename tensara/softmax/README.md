---
title: Softmax
platform: Tensara
upstream: softmax
url: https://tensara.org/problems/softmax
difficulty: medium
tags: [softmax, online-softmax]
status: solved
---

# Softmax

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/softmax)

## Problem
Softmax along an arbitrary dimension.

## Approach
Online softmax over the `(outer, R, inner)` view: each lane or thread keeps a
running `(max, Σe^{x−max})` pair, pairs merge with rescaling, and a second
pass writes `e^{x−max}/Σ`. That is two reads and one write per element.

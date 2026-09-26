---
title: Hinge Loss
platform: Tensara
upstream: hinge-loss
url: https://tensara.org/problems/hinge-loss
difficulty: easy
tags: [loss, elementwise]
status: solved
---

# Hinge Loss

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/hinge-loss)

## Problem
Elementwise `max(0, 1 − p·t)`.

## Approach
Grid-stride elementwise kernel.

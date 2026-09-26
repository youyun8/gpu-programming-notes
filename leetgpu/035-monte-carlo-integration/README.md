---
title: Monte Carlo Integration
platform: LeetGPU
upstream: medium/35_monte_carlo_integration
url: https://leetgpu.com/challenges/monte-carlo-integration
difficulty: medium
tags: [reduction]
status: solved
---

# Monte Carlo Integration

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/monte-carlo-integration)

## Problem
`(b - a) · mean(y)`.

## Approach
A two-pass sum reduction (`float4` loads, fp64 block partials); the final block
scales by `(b-a)/n`.

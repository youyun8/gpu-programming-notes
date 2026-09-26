---
title: Logistic Regression
platform: LeetGPU
upstream: medium/34_logistic_regression
url: https://leetgpu.com/challenges/logistic-regression
difficulty: medium
tags: [optimization, newton, irls, cholesky]
status: solved
---

# Logistic Regression

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/logistic-regression)

## Problem
Maximum-likelihood logistic regression coefficients.

## Approach
The same Newton–Raphson (IRLS) iteration as the reference, in fp64:
`p = σ(Xβ)`, `W = max(p(1-p), 1e-8)`, `g = Xᵀ(p-y) + λβ`,
`H = XᵀWX + λI`, `β -= H⁻¹g`, until `‖step‖ < 1e-8`.

Each iteration launches four kernels: per-sample terms (warp per sample), a
tiled weighted Gram matrix, the gradient, and a single-block Cholesky solve
that also updates `β` and computes `‖step‖²`. Only that one scalar is copied
back to the host to decide whether to stop. Newton converges in about 10
iterations, while plain gradient descent would need thousands.

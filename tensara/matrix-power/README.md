---
title: Matrix Nth Power
platform: Tensara
upstream: matrix-power
url: https://tensara.org/problems/matrix-power
difficulty: medium
tags: [gemm, binary-exponentiation]
status: solved
---

# Matrix Nth Power

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/matrix-power)

## Problem
`Aⁿ` for a square matrix (`n = 0` gives the identity).

## Approach
Binary exponentiation in the same multiplication order as
`torch.linalg.matrix_power`, using the register-blocked SGEMM (shared with the
[LeetGPU version](../../leetgpu/037-matrix-power)).

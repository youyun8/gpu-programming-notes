---
title: Matrix Power
platform: LeetGPU
upstream: medium/37_matrix_power
url: https://leetgpu.com/challenges/matrix-power
difficulty: medium
tags: [gemm, binary-exponentiation]
status: solved
---

# Matrix Power

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/matrix-power)

## Problem
`A^P` for `N ≤ 1024`, `P ≤ 20`.

## Approach
Binary exponentiation needs `O(log P)` GEMMs instead of `P-1`. The products
are done **in the same order as `torch.linalg.matrix_power`** (special cases
for `P ≤ 3`; otherwise walk the bits of `P` with `result = result @ z`), so
the rounding matches the reference closely. Each product uses the 64×64
register-blocked SGEMM, with ping-pong scratch buffers.

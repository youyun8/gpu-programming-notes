---
title: 2D Jacobi Stencil
platform: LeetGPU
upstream: medium/69_jacobi_stencil_2d
url: https://leetgpu.com/challenges/2d-jacobi-stencil
difficulty: medium
tags: [stencil]
status: solved
---

# 2D Jacobi Stencil

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/2d-jacobi-stencil)

## Problem
One 5-point Jacobi sweep; boundary cells are copied.

## Approach
A 32×8 thread block with `threadIdx.x` along columns: the left, right and
centre reads of a warp share cache lines, and the rows above and below are
reused from L2, so the kernel runs close to copy bandwidth. The neighbour sum
uses the reference's summation order.

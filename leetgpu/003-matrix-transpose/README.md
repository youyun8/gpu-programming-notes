---
title: Matrix Transpose
platform: LeetGPU
upstream: easy/3_matrix_transpose
url: https://leetgpu.com/challenges/matrix-transpose
difficulty: easy
tags: [shared-memory, coalescing, bank-conflicts]
status: solved
---

# Matrix Transpose

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/matrix-transpose)

## Problem
Write `output (cols×rows) = inputᵀ` for a row-major float32 `input (rows×cols)`.

## Approach
A naive transpose has either uncoalesced reads or uncoalesced writes. Stage
a 32×32 tile in shared memory: read rows of the input (coalesced), then write
rows of the output (coalesced) by swapping the block coordinates. The tile is
declared `[32][33]`; the extra column shifts each row by one bank so the
column-wise shared reads are conflict-free. A 32×8 block moves 4 elements per
thread.

## Pitfalls
- Bounds must be checked separately for the read (`rows × cols`) and the write
  (`cols × rows`) phase.

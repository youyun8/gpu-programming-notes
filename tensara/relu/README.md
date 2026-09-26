---
title: ReLU
platform: Tensara
upstream: relu
url: https://tensara.org/problems/relu
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# ReLU

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/relu)

## Problem

Apply ReLU elementwise to an $M\times N$ float32 matrix, matching
`torch.relu`. Test matrices go from $4096\times4096$ to $8192\times8192$ (up to 67 M elements). The check is `rtol = 6e-5`, `atol = 3e-5`. This is the
simplest possible bandwidth benchmark: the arithmetic is a single `max`.

## Formulation

$$
C_{ij} = \operatorname{ReLU}(A_{ij}) = \max(A_{ij}, 0)
$$

| Symbol | Meaning |
|---|---|
| $A$ | Input matrix, $M\times N$ float32, row-major |
| $C$ | Output matrix, same shape |
| $\operatorname{ReLU}$ | Rectified linear unit |

## Approach

All Tensara elementwise problems share one kernel shape:

1. **`float4` grid-stride loop.** The buffer is viewed as $\lfloor n/4 \rfloor$
   16-byte vectors; each iteration loads one `float4`, applies the scalar
   function to the four lanes, and stores one `float4`. `cudaMalloc`
   returns 256-byte-aligned pointers, so the reinterpretation is safe.
2. **Scalar tail** for the last $n \bmod 4$ elements.
3. **Launch** 256-thread blocks, capped at 4096 blocks; the grid-stride
   loop covers any size, and 4096 × 256 threads are enough to saturate DRAM.
4. The function is a `__forceinline__` device function, so the loop body is
   branch-free apart from the select in the function itself.

`fmaxf(x, 0.0f)` is one instruction. The whole kernel is a memory copy with a filter, so its speed is set entirely by how well the loads and stores use DRAM: 16-byte accesses, enough bytes in flight (4096 × 256 threads × 16 B = 16 MB outstanding at most), and no redundant traffic.

## Cost Analysis

$$
n = MN, \qquad Q = 8\,n\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $n$ | Number of elements |
| $Q$ | Compulsory DRAM traffic: read the input(s) once, write the output once |
| $\beta$ | DRAM bandwidth (about 2–3 TB/s on current data-centre GPUs) |
| $T_{\min}$ | Bandwidth lower bound on the kernel time |

For $8192\times8192$: $Q = 537$ MB, about 0.27 ms at 2 TB/s. The best achievable is typically 85–92 % of the nominal bandwidth.

## Pitfalls

- **NaN handling**: `fmaxf(NaN, 0) = 0`, while `torch.relu(NaN) = NaN`.
  Test data has no NaNs; write `x > 0 ? x : 0` (which also returns 0 for
  NaN) or `x < 0 ? 0 : x` (which propagates NaN) if it matters.
- The launch parameters `n, m` are rows and columns; only the product is
  used.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Leaky ReLU](../leaky-relu/), [Vector Addition](../vector-addition/), LeetGPU [ReLU](../../leetgpu/021-relu/).

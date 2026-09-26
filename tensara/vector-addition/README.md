---
title: Vector Addition
platform: Tensara
upstream: vector-addition
url: https://tensara.org/problems/vector-addition
difficulty: easy
tags: [elementwise, float4]
status: solved
---

# Vector Addition

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/vector-addition)

## Problem

Add two float32 vectors of length $n$, from $2^{20}$ up to $2^{30}$
elements (4 GB per vector at the top end). The check is `rtol = 2e-4`,
`atol = 1e-4`. It is the "hello world" of CUDA, but at $2^{30}$ elements
64-bit indexing and a sane grid size matter.

## Formulation

$$
c_i = a_i + b_i, \qquad 0 \le i < n
$$

| Symbol | Meaning |
|---|---|
| $a, b$ | Input vectors (`d_input1`, `d_input2`) |
| $c$ | Output vector (`d_output`) |
| $n$ | Vector length |

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

With $n = 2^{30}$, a one-thread-per-element launch would need $2^{22}$ blocks of 256 threads: legal (the $x$ grid limit is $2^{31} - 1$) but wasteful. The capped grid-stride loop uses 4096 blocks and every index is `size_t`, because $4n$ bytes $= 2^{32}$ overflows 32-bit arithmetic.

## Cost Analysis

$$
n = MN, \qquad Q = 12\,n\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $n$ | Number of elements |
| $Q$ | Compulsory DRAM traffic: read the input(s) once, write the output once |
| $\beta$ | DRAM bandwidth (about 2–3 TB/s on current data-centre GPUs) |
| $T_{\min}$ | Bandwidth lower bound on the kernel time |

For $n = 2^{30}$: $Q = 12.9$ GB, about 6.4 ms at 2 TB/s. The intensity is $1/12$ flop per byte.

## Pitfalls

- **64-bit indices**: `int i = blockIdx.x * blockDim.x + threadIdx.x`
  overflows at $2^{31}$.
- **Three streams**, so $Q = 12n$, not $8n$.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Matrix Scalar](../matrix-scalar/), [ReLU](../relu/), LeetGPU [Vector Addition](../../leetgpu/001-vector-add/).

---
title: Matrix Addition
platform: LeetGPU
upstream: easy/8_matrix_addition
url: https://leetgpu.com/challenges/matrix-addition
difficulty: easy
tags: [elementwise, vectorized, memory-bound]
status: solved
---

# Matrix Addition

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/matrix-addition)

## Problem

Add two $N \times N$ float32 matrices element by element into `C`
($1 \le N \le 4096$; benchmark $N = 4096$). The matrices are stored
contiguously in row-major order, so the 2-D structure is irrelevant to the
computation. This problem shows how to exploit that fact with **vectorised
128-bit accesses**.

## Formulation

$$
C_{rc} = A_{rc} + B_{rc}, \qquad 0 \le r, c < N
\quad\Longleftrightarrow\quad
C_k = A_k + B_k, \qquad k = rN + c,\ \ 0 \le k < N^2
$$

| Symbol | Meaning |
|---|---|
| $N$ | matrix side length |
| $r,\ c$ | row and column index |
| $k$ | flattened row-major index |
| $A,\ B$ | input matrices (float32) |
| $C$ | output matrix (float32) |

### Vectorised Split

$$
N^2 = 4V + R, \qquad V = \left\lfloor \frac{N^2}{4} \right\rfloor, \quad R = N^2 \bmod 4
$$

| Symbol | Meaning |
|---|---|
| $V$ | number of complete `float4` groups (4 consecutive floats) |
| $R$ | leftover scalar elements at the end, $0 \le R \le 3$ |

## Approach

- Thread $t$ (global index) with $t < V$ loads `float4` number $t$ of `A`
  and of `B`, adds the four lanes, and stores one `float4` to `C`. That is 3
  memory instructions for 4 elements instead of 12.
- Threads $t < R$ additionally handle the scalar tail element $4V + t$.
- Grid: $\lceil (V + 1)/256 \rceil$ blocks of 256 threads, which covers both
  the vector part and the tail threads.

Vector loads reduce the number of load/store instructions and in-flight
memory requests. This helps the memory system reach peak bandwidth with fewer
warps. The DRAM traffic is the same as for a scalar kernel.

## Cost Analysis

$$
Q = 3 \cdot 4N^2 = 12N^2 \ \text{bytes}, \qquad W = N^2, \qquad I = \frac{1}{12}, \qquad T_{\min} = \frac{12N^2}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM traffic (read $A$, $B$; write $C$) |
| $W$ | additions |
| $I$ | arithmetic intensity (FLOP/byte) |
| $\beta$ | DRAM bandwidth |

$N = 4096$ gives $Q = 201$ MB, so $T_{\min} \approx 100\ \mu s$ at 2 TB/s.

## Pitfalls

- **Treating it as 2-D.** A 2-D grid with `C[r][c]` indexing is correct but
  adds integer work and makes vectorisation awkward. The flat view is simpler
  and faster.
- **Tail handling.** $N^2$ is a multiple of 4 whenever $N$ is even. For odd
  $N$ the last 1–3 elements need the scalar path.
- **Alignment.** `float4` requires 16-byte alignment, which `cudaMalloc`
  guarantees.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-5`,
including odd $N$, where the tail path is exercised.

## Related

- [Vector Addition](../001-vector-add/), [Matrix Copy](../031-matrix-copy/).
- Tensara [Vector Addition](../../tensara/vector-addition/), [Matrix-Scalar](../../tensara/matrix-scalar/).

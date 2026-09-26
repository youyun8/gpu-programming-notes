---
title: All-Pairs Shortest Path
platform: Tensara
upstream: all-pairs-shortest-path
url: https://tensara.org/problems/all-pairs-shortest-path
difficulty: medium
tags: [graph, floyd-warshall, blocked-algorithm, shared-memory]
status: solved
---

# All-Pairs Shortest Path

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/all-pairs-shortest-path)

## Problem

All-pairs shortest paths on a dense weighted digraph given as an $n\times n$
adjacency matrix. Positive integer weights; **0 means "no edge"** (except on
the diagonal); unreachable pairs must be reported as $-1$. Test sizes are
$n = 512 \dots 4096$, and the check is `rtol = 1e-4`, `atol = 1e-3`. The
algorithm is Floyd–Warshall, as in the reference. The encoding of "no edge"
and "unreachable" is what differs from the
[LeetGPU version](../../leetgpu/073-all-pairs-shortest-paths/).

## Formulation

$$
d^{(0)}_{ij} = \begin{cases} 0, & i = j \\ +\infty, & a_{ij} = 0 \\ a_{ij}, & \text{otherwise}\end{cases}, \qquad
d^{(k+1)}_{ij} = \min\bigl(d^{(k)}_{ij},\ d^{(k)}_{ik} + d^{(k)}_{kj}\bigr), \qquad
\text{out}_{ij} = \begin{cases} d^{(n)}_{ij}, & d^{(n)}_{ij} < \infty \\ -1, & \text{otherwise}\end{cases}
$$

| Symbol | Meaning |
|---|---|
| $n$ | number of vertices |
| $a_{ij}$ | input weight of edge $i \to j$ (0 = absent) |
| $d^{(k)}_{ij}$ | shortest $i\to j$ distance using intermediates $< k$ |
| $\text{out}_{ij}$ | output; $-1$ for unreachable pairs |

## Approach

1. **`prepare`**: grid-stride map. The diagonal becomes 0, zeros become
   $+\infty$, and everything else is copied into `output`.
2. **Blocked Floyd–Warshall** with 32 × 32 tiles, in place in `output`. Each
   round $b$ runs three kernels: the diagonal tile (32 dependent steps in
   shared memory), the row/column panels, and all remaining tiles
   ($(\min, +)$ products of the two panels, no inner barriers). The full
   derivation is on the [LeetGPU page](../../leetgpu/073-all-pairs-shortest-paths/).
3. **`unreachableToMinusOne`**: $+\infty \to -1$.

## Cost Analysis

$$
W = 2n^3, \qquad Q \approx \frac{n}{32}\cdot 3\cdot 4n^2\ \text{bytes}, \qquad \#\text{launches} = 3\left\lceil\frac{n}{32}\right\rceil + 2
$$

| Symbol | Meaning |
|---|---|
| $W$ | add and min operations |
| $Q$ | DRAM bytes: each round reads and writes every tile once, plus its two panels |
| #launches | kernel launches (384 at $n = 4096$) |

At $n = 4096$: $W = 1.4\times10^{11}$ operations and $Q \approx 25$ GB, so the
kernel is compute-bound in phase 3. Register tiling of phase 3 (several
outputs per thread) is the main remaining optimisation.

## Pitfalls

- **"0 = no edge"** only off the diagonal; the diagonal must be 0.
- **Output encoding** $-1$ for $+\infty$.
- **Integer weights** make all path lengths exact in float32 up to $2^{24}$.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- LeetGPU [All-Pairs Shortest Paths](../../leetgpu/073-all-pairs-shortest-paths/),
  [Shortest Path](../shortest-path/), [Min Spanning Tree](../min-spanning-tree/).

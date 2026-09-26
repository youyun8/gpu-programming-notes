---
title: Single Source Shortest Path
platform: Tensara
upstream: shortest-path
url: https://tensara.org/problems/shortest-path
difficulty: medium
tags: [graph, bellman-ford, early-exit]
status: solved
---

# Single Source Shortest Path

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/shortest-path)

## Problem

Single-source shortest paths in a directed graph of $N$ vertices
($N$ = 512 … 8192) given as a dense adjacency matrix with positive integer
weights (0 means no edge) and a source $s$. Unreachable vertices get
$-1$. The reference runs $N - 1$ Bellman–Ford sweeps. The check is
`rtol = 1e-4`, `atol = 1e-3`.

## Formulation

$$
d^{(0)}[v] = \begin{cases} 0, & v = s \\ +\infty, & v \ne s \end{cases}, \qquad
d^{(t+1)}[v] = \min\Bigl(d^{(t)}[v],\ \min_{u\,:\,a_{uv} > 0} \bigl(d^{(t)}[u] + a_{uv}\bigr)\Bigr)
$$

$$
\text{out}[v] = \begin{cases} d^{(T)}[v], & d^{(T)}[v] < \infty \\ -1, & \text{otherwise} \end{cases}
$$

| Symbol | Meaning |
|---|---|
| $s$ | source vertex |
| $a_{uv}$ | weight of edge $u \to v$ (0 = absent) |
| $d^{(t)}[v]$ | shortest distance to $v$ using at most $t$ edges |
| $T$ | number of sweeps performed |
| out | final distances, $-1$ if unreachable |

After $t$ sweeps $d^{(t)}$ is exact for every vertex whose shortest path
has at most $t$ edges, so the iteration can stop at the first sweep that
changes nothing:

$$
d^{(t+1)} = d^{(t)} \ \Longrightarrow\ d^{(t)} = d^{(\infty)}, \qquad T \le \text{(max edges on a shortest path)} + 1
$$

| Symbol | Meaning |
|---|---|
| $d^{(\infty)}$ | the true shortest distances (reached after at most $N - 1$ sweeps) |

## Approach

1. **`initDist`**: $0$ at the source, $+\infty$ elsewhere.
2. **`relax`**, one thread per destination $v$: loop over all $u$ and read
   $a_{uv}$, i.e. column $v$. For a fixed $u$ the 32 lanes of a warp read
   $a_{u,v..v+31}$: 128 contiguous bytes, coalesced. The new distances go
   to a second buffer (Jacobi style), and any change sets a device flag.
3. **Host loop**: reset the flag, launch, read the flag back, swap
   buffers, stop when nothing changed (at most $N - 1$ sweeps).
4. **`finish`**: $+\infty \to -1$.

## Cost Analysis

$$
Q = 4N^2 \cdot T\ \text{bytes}, \qquad W = N^2 T\ \text{relaxations}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes: the whole matrix per sweep |
| $W$ | add-and-compare operations |
| $T$ | sweeps until convergence |

At $N = 8192$ the matrix is 268 MB, about 0.13 ms per sweep at 2 TB/s. For
random dense graphs shortest paths have few edges, so $T$ is small; the
reference's fixed $N - 1$ sweeps would be over a second. For $N$ threads
= 8192, only 32 blocks run, so each sweep is latency-bound; splitting the
$u$ loop across several threads per $v$ (and a min-reduction) would use
more of the GPU.

## Pitfalls

- **Read the flag every sweep**: the device→host copy synchronizes, which
  costs a few µs per sweep but enables the early exit.
- **Jacobi vs Gauss–Seidel**: updating in place is also correct for
  Bellman–Ford (it can only converge faster), but double buffering keeps
  the kernel free of read/write races.
- **0 = no edge**; **unreachable = $-1$**.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [All-Pairs Shortest Path](../all-pairs-shortest-path/), [Min Spanning Tree](../min-spanning-tree/),
  LeetGPU [BFS Shortest Path](../../leetgpu/046-bfs-shortest-path/).

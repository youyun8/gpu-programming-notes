---
title: Minimum Spanning Tree
platform: Tensara
upstream: min-spanning-tree
url: https://tensara.org/problems/min-spanning-tree
difficulty: medium
tags: [graph, prim, single-block, reduction]
status: solved
---

# Minimum Spanning Tree

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/min-spanning-tree)

## Problem

Total weight of a minimum spanning tree of an undirected graph with $n$
vertices ($n$ = 1024 … 6144), given as a symmetric dense adjacency matrix
of positive integer weights where 0 means "no edge". The reference runs
Prim's algorithm and returns $+\infty$ for a disconnected graph (the
statement says $-1$; the reference wins). The check is `rtol = 1e-4`,
`atol = 1e-3`.

## Formulation

$$
\text{MST} = \operatorname*{arg\,min}_{T \in \mathcal{T}(G)} \sum_{(u,v)\in T} a_{uv}, \qquad \text{output} = \sum_{(u,v)\in\text{MST}} a_{uv}
$$

| Symbol | Meaning |
|---|---|
| $G$ | The graph; an edge $(u, v)$ exists iff $a_{uv} > 0$ |
| $a_{uv}$ | Edge weight (symmetric, $a_{uv} = a_{vu}$) |
| $\mathcal{T}(G)$ | The set of spanning trees of $G$ ($n - 1$ edges connecting all vertices) |
| MST | A tree of minimum total weight |

Prim's algorithm grows a tree $S$ from vertex 0, keeping for every vertex
outside $S$ the lightest edge into $S$:

$$
\text{best}[v] = \min_{u \in S} a_{uv}, \qquad
u^\star = \operatorname*{arg\,min}_{v \notin S} \text{best}[v], \qquad
S \leftarrow S \cup \{u^\star\}, \qquad \text{best}[v] \leftarrow \min(\text{best}[v],\ a_{u^\star v})
$$

| Symbol | Meaning |
|---|---|
| $S$ | Vertices already in the tree (`in_tree`) |
| $\text{best}[v]$ | Weight of the lightest edge from $v$ to $S$ ($+\infty$ if none) |
| $u^\star$ | Vertex added in this step (ties → smallest index) |

Each step adds $\text{best}[u^\star]$ to the total. By the cut property, the
lightest edge crossing $(S, V\setminus S)$ is always in some MST.

## Approach

**One block of 1024 threads runs all $n - 1$ steps**, synchronizing with
`__syncthreads()` instead of kernel launches:

1. **Arg-min**: every thread scans its strided vertices outside $S$; a
   warp shuffle and then a 32-entry shared-memory pass select
   $(\text{best}[u^\star], u^\star)$ with lowest-index tie-breaking.
2. **Stop** with $+\infty$ if the minimum is infinite (disconnected).
3. **Relax**: threads read row $u^\star$ of the matrix (coalesced, $n$ floats)
   and lower $\text{best}[v]$.
4. The total is accumulated in `double` by every thread (they all see the
   same values) and thread 0 writes it.

## Cost Analysis

$$
W = O(n^2), \qquad Q = 4n^2\ \text{bytes (each row read once)}, \qquad
T \approx (n - 1)\bigl(t_{\text{sync}} + t_{\text{scan}}(n)\bigr)
$$

| Symbol | Meaning |
|---|---|
| $W$ | Total work: $n$ steps of $O(n)$ |
| $Q$ | DRAM bytes: row $u^\star$ once per step |
| $t_{\text{sync}}$ | Cost of two block barriers and the arg-min (~µs) |
| $t_{\text{scan}}(n)$ | Time to scan $n$ values with 1024 threads |

At $n = 6144$: 151 MB of rows and ~6 K steps of a few µs each, so the run
time (~10–20 ms) is dominated by the serial step latency, not bandwidth.
A single block uses one SM; the alternative, one launch per step across
the whole GPU, pays ~3–5 µs of launch overhead per step instead. Borůvka's
algorithm ($O(\log n)$ parallel rounds) is the way to use the whole GPU.

## Pitfalls

- **0 means "no edge"**, so it must become $+\infty$ in `best`, not a free
  edge of weight 0.
- **Disconnected graph**: return $+\infty$, as the reference does.
- **Precision**: integer weights summed in fp32 lose exactness above
  $2^{24}$; the fp64 total stays exact.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Shortest Path](../shortest-path/), [All-Pairs Shortest Path](../all-pairs-shortest-path/),
  [Argmin](../argmin/).

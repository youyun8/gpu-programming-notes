---
title: BFS Shortest Path
platform: LeetGPU
upstream: hard/46_bfs_shortest_path
url: https://leetgpu.com/challenges/bfs-shortest-path
difficulty: hard
tags: [graph, bfs, persistent-kernel, atomics]
status: solved
---

# BFS Shortest Path

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/bfs-shortest-path)

## Problem

Shortest path length (number of 4-neighbour moves) between two free cells
of a `rows × cols` grid with obstacles, or $-1$ if unreachable
($\text{rows}, \text{cols} \le 1000$; benchmark $500\times500$). The answer
must match exactly. BFS is irregular (the frontier size varies wildly), it
is sequential across levels, and a maze can have $O(\text{rows}\cdot\text{cols})$
levels. All three are hard for GPUs.

## Formulation

Model the grid as a graph $G = (V, E)$ with the free cells as vertices and
4-neighbour adjacency as edges. BFS computes distances level by level:

$$
F_0 = \{s\}, \qquad
F_{\ell+1} = \bigl\{\, v \in V \setminus (F_0 \cup \dots \cup F_\ell) \ :\ \exists\, u \in F_\ell,\ (u, v) \in E \,\bigr\}, \qquad
d(s, t) = \min\{\ell : t \in F_\ell\}
$$

| Symbol | Meaning |
|---|---|
| $V$ | free cells (grid value 0), indexed $r\cdot\text{cols} + c$ |
| $E$ | pairs of free cells that differ by one step up/down/left/right |
| $s,\ t$ | start and goal cells |
| $F_\ell$ | frontier: cells at distance exactly $\ell$ from $s$ |
| $d(s,t)$ | shortest-path length; $-1$ if $t$ is never reached (frontier becomes empty) |

Each cell enters exactly one frontier, so the total work is
$O(\lvert V\rvert + \lvert E\rvert) = O(\text{rows}\cdot\text{cols})$.

## Approach

### Level-synchronous BFS in one persistent block

- A single block of 1024 threads runs the whole search. Two global arrays
  hold the current and next frontier, and a global `visited` map holds one
  int per cell.
- Per level:
  1. Threads grid-stride over the current frontier. For each cell they
     compute its up-to-4 in-bounds neighbours and skip obstacles.
  2. `atomicCAS(&visited[nb], 0, 1) == 0` **claims** a neighbour. Exactly one
     thread wins per cell, so it is appended exactly once, through
     `atomicAdd` on the shared counter `s_next_size`.
  3. If the claimed cell is the goal, `s_found = 1`.
  4. `__syncthreads()`, swap the queues, increment the level, then
     `__syncthreads()` again.
- The loop stops when the goal is found or the frontier is empty.

### Why one block?

Separating levels needs a barrier. Across the whole GPU that means one kernel
launch per level (~3–5 µs each) or a cooperative-groups grid sync. A
$500 \times 500$ serpentine maze has about 125 000 levels, which would take
over 0.5 s in launch overhead alone. Inside one block the barrier is a
`__syncthreads()` of a few tens of nanoseconds. Only one SM is used, but for
sparse frontiers (usually tens to hundreds of cells) one SM is not the
bottleneck.

## Cost analysis

$$
W = O(\lvert V\rvert), \qquad T \approx L\cdot t_{\text{level}} + \frac{4\lvert V\rvert}{\text{throughput}_{\text{SM}}}
$$

| Symbol | Meaning |
|---|---|
| $W$ | total work: each cell is expanded once, with 4 neighbour checks |
| $L$ | number of BFS levels ($= d(s,t)$ or the eccentricity of $s$) |
| $t_{\text{level}}$ | fixed cost per level (two barriers plus shared-memory bookkeeping) |
| throughput$_{\text{SM}}$ | neighbour checks per second on one SM, limited by atomics and global-memory latency |

Open grids have $L \approx$ rows + cols and wide frontiers. Mazes have huge
$L$ and narrow frontiers. The persistent block handles both without
host-side involvement.

## Pitfalls

- **Visited check without atomics.** Two threads can both see
  `visited == 0` and enqueue the same cell twice. The result stays correct
  but the work can blow up.
- **$s = t$** returns 0 without searching.
- **Early exit.** The goal can be detected when it is *claimed*, one level
  earlier than when it would be expanded. The level counter is incremented
  after the barrier, so the reported distance is exact.

## Verification

Exact match on all LeetGPU cases in [cuemu](../../tools/cuemu/README.md),
including unreachable goals, $s = t$, $1 \times 1$ grids, and long
serpentine mazes.

## Related

- [All-Pairs Shortest Paths](../073-all-pairs-shortest-paths/), Tensara [Shortest Path](../../tensara/shortest-path/),
  [Stream Compaction](../072-stream-compaction/) (building frontiers with scans instead of atomics).

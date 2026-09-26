---
title: Stream Compaction
platform: LeetGPU
upstream: medium/72_stream_compaction
url: https://leetgpu.com/challenges/stream-compaction
difficulty: medium
tags: [scan, compaction, filter]
status: solved
---

# Stream Compaction

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/stream-compaction)

## Problem

Stable **stream compaction**: copy every positive element of $A$ (length
$N \le 10^8$) to the front of `out`, preserving order, and fill the rest with
0 (benchmark $N = 5\times10^7$; exact). Compaction ("filter") builds work
lists on the GPU: active rays, surviving particles, non-zero entries, BFS
frontiers.

## Formulation

$$
p_i = [\,A_i > 0\,], \qquad o_i = \sum_{j < i} p_j \ \ (\text{exclusive scan}), \qquad
\text{out}_{o_i} = A_i\ \ \text{whenever } p_i = 1, \qquad
\text{out}_{k..N-1} = 0,\ \ k = \sum_j p_j
$$

| Symbol | Meaning |
|---|---|
| $N$ | Input length |
| $A_i$ | Input value |
| $p_i$ | Predicate (1 if kept) |
| $o_i$ | Output slot of element $i$: the number of kept elements before it |
| $k$ | Total number of kept elements |
| out | Compacted output |

The exclusive scan of the predicate gives each kept element a **unique,
order-preserving** destination. That is exactly the stability requirement.

## Approach

Reduce-then-scan over 2048-element chunks (256 threads × 8 items):

1. **`chunkCounts`**: count predicates per chunk (block reduction).
2. **`scanCounts`** (1 block): exclusive scan of the chunk counts into chunk
   offsets. It also produces the total $k$ in device memory.
3. **`scatter`**: each thread counts its 8 consecutive items. A block
   exclusive scan of those counts, plus the chunk offset, gives the thread's
   first output slot. The thread then walks its 8 items and writes the kept
   ones consecutively.
4. **Zero fill**: positions $[k, N)$ are set to 0. The harness claims `out`
   is pre-initialised, but zero-filling makes the kernel independent of that.

## Cost Analysis

$$
Q = \underbrace{4N}_{\text{count}} + \underbrace{4N}_{\text{scatter read}} + \underbrace{4k + 4(N-k)}_{\text{writes}} = 12N\ \text{bytes}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes |
| $k$ | Number of kept elements |

Benchmark: 600 MB, i.e. ≈ 0.3 ms at 2 TB/s. The scatter writes are
contiguous per thread but not across the warp (each thread writes a variable
number of elements), so write coalescing is partial. A warp-level variant
uses `__ballot_sync` and `__popc` to give every lane its slot inside the
warp, and writes 32 consecutive kept values per instruction.

## Pitfalls

- **Zero is not positive.** `A[i] = 0.0` is dropped (`> 0`, not `>= 0`).
- **Stability.** Atomics (`atomicAdd` on a global cursor) are simpler but
  reorder the output. The reference requires the original order.

## Verification

Exact match on all LeetGPU cases in [cuemu](../../tools/cuemu/README.md),
including all-positive, all-non-positive, and alternating inputs.

## Related

- [Prefix Sum](../016-prefix-sum/), [Segmented Prefix Sum](../070-segmented-prefix-sum/),
  [Top-K](../029-top-k-selection/) (gather by threshold).

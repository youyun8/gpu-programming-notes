---
title: Histogramming
platform: LeetGPU
upstream: medium/13_histogramming
url: https://leetgpu.com/challenges/histogramming
difficulty: medium
tags: [histogram, atomics, privatization, shared-memory]
status: solved
---

# Histogramming

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/histogramming)

## Problem

Count how many times each integer value $v \in [0, B)$ occurs in an int32
array of length $N$ ($1 \le N \le 10^8$, $1 \le B \le 1024$; benchmark
$N = 5\times10^7$, $B = 256$). The output is an int32 array of $B$ counts
and must match exactly. Histograms are the textbook case of **write
contention**: many threads want to increment the same few counters at the
same time.

## Formulation

$$
h_v = \sum_{i=0}^{N-1} [\,x_i = v\,], \qquad 0 \le v < B
$$

| Symbol | Meaning |
|---|---|
| $N$ | Number of input values |
| $B$ | Number of bins (`num_bins`) |
| $x_i$ | $i$-th input value, $0 \le x_i < B$ |
| $h_v$ | Count for bin $v$ |
| $[\cdot]$ | Iverson bracket: 1 if the condition holds, else 0 |

### Privatisation

Split the input among $G$ blocks, count each part separately, and add the
partial histograms:

$$
h_v = \sum_{g=0}^{G-1} h^{(g)}_v, \qquad h^{(g)}_v = \sum_{i \in \mathcal{P}_g} [\,x_i = v\,]
$$

| Symbol | Meaning |
|---|---|
| $G$ | Number of blocks ($\le 1024$) |
| $\mathcal{P}_g$ | Indices processed by block $g$ (its grid-stride slice) |
| $h^{(g)}_v$ | Block $g$'s private count for bin $v$, kept in shared memory |

## Approach

1. `cudaMemset(histogram, 0, …)`. The output buffer is not zeroed by the
   harness.
2. **Private histogram.** Each block zeroes `s_hist[B]` in shared memory, then
   runs a grid-stride loop doing `atomicAdd(&s_hist[x_i], 1)`. Shared-memory
   atomics execute in the SM, cost a few cycles, and contend only with the
   256 threads of the same block.
3. **Merge.** After `__syncthreads()`, thread $t$ adds
   $h^{(g)}_t, h^{(g)}_{t+256}, \dots$ to global memory with `atomicAdd`,
   skipping zeros.

### Why It Is Faster

| Variant | Global atomics | Contention |
|---|---|---|
| Naive: one global atomic per element | $N = 5\times10^7$ | Every thread in the GPU on 256 addresses |
| Privatised | $\le G \cdot B = 262\,144$ | Only 1024 blocks per address, once each |

Global atomics are resolved in L2 at a fixed throughput per address, so the
naive version is serialised on hot bins. Privatisation moves more than 99% of
the updates into shared memory.

## Cost Analysis

$$
Q \approx 4N + 4GB \ \text{bytes}, \qquad T_{\min} = \frac{4N}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM traffic: read every input once, plus the merge of $G$ partial histograms |
| $\beta$ | DRAM bandwidth |
| $T_{\min}$ | Bandwidth lower bound (reading the input) |

At the benchmark size, $Q \approx 200$ MB, i.e. $\approx 100\ \mu s$ at 2 TB/s. In
practice, throughput depends on the data distribution. If all values fall in
one bin, the shared-memory atomics serialise within the block, and
warp-aggregated atomics (`__match_any_sync` + one add per group) or
per-warp sub-histograms help.

## Pitfalls

- **Not zeroing the output.** The results are garbage and differ between runs.
- **Shared array size.** It is sized for the maximum $B = 1024$. Only the
  first $B$ entries are used, and every loop is bounded by $B$.
- **Out-of-range values.** The constraints guarantee $0 \le x_i < B$. The
  kernel still checks, mirroring the reference's mask.

## Verification

Exact integer match on all LeetGPU cases in
[cuemu](../../tools/cuemu/README.md), including $B = 1$ (every element in one
bin, maximum contention).

## Related

- [Count Array Element](../043-count-array-element/), [Top-K](../029-top-k-selection/),
  [Radix Sort](../036-radix-sort/) (whose digit counting is a histogram).
- Tensara [Histogram](../../tensara/histogram/).

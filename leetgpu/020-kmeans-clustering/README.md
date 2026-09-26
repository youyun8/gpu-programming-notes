---
title: K-Means Clustering
platform: LeetGPU
upstream: hard/20_kmeans_clustering
url: https://leetgpu.com/challenges/k-means-clustering
difficulty: hard
tags: [clustering, atomics, privatization, iterative]
status: solved
---

# K-Means Clustering

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/k-means-clustering)

## Problem

Run $T$ iterations of **Lloyd's k-means** on $n$ 2-D points with $k$
clusters ($1 \le n \le 10^6$, $1 \le k \le 1000$; benchmark $k = 5$, $T = 30$,
$n = 10\,000$). Starting from the given initial centroids, each iteration
assigns every point to its nearest centroid, then moves each centroid to the
mean of its points. Outputs are the final centroids and the labels of the
last assignment, with tolerance `1e-4`.

## Formulation

For iteration $t = 0, \dots, T-1$:

$$
\ell_i^{(t)} = \arg\min_{0 \le c < k}\ \bigl(x_i - \mu^{(t)}_{c,x}\bigr)^2 + \bigl(y_i - \mu^{(t)}_{c,y}\bigr)^2
$$

$$
\mu^{(t+1)}_c =
\begin{cases}
\dfrac{1}{\lvert S_c\rvert}\displaystyle\sum_{i \in S_c} (x_i, y_i), & \lvert S_c\rvert > 0\\[2mm]
\mu^{(t)}_c, & \lvert S_c\rvert = 0
\end{cases}
\qquad S_c = \{\, i : \ell^{(t)}_i = c \,\}
$$

| Symbol | Meaning |
|---|---|
| $n$ | Number of points (`sample_size`) |
| $k$ | Number of clusters |
| $T$ | Number of iterations (`max_iterations`) |
| $(x_i, y_i)$ | Coordinates of point $i$ |
| $\mu^{(t)}_c = (\mu_{c,x}, \mu_{c,y})$ | Centroid $c$ at iteration $t$; $\mu^{(0)}$ = the initial centroids |
| $\ell^{(t)}_i$ | Label (cluster index) of point $i$; ties go to the smallest $c$ |
| $S_c$ | Set of points assigned to cluster $c$ |
| $\lvert S_c\rvert$ | Cluster size; empty clusters keep their centroid |

## Approach

Each iteration has two kernels. The iteration loop runs on the host; the
data never leaves the GPU.

### 1. `assignPoints` (≤ 1024 Blocks × 256 Threads)

- Stage all $k$ centroids in shared memory. Every point compares against
  every centroid, so the centroid reads are broadcasts.
- A grid-stride loop over points computes $d = (x-\mu_x)^2 + (y-\mu_y)^2$
  with `__fsub_rn`, `__fmul_rn`, `__fadd_rn`, so **no FMA contraction**
  happens. This matches PyTorch's `expanded_x**2 + expanded_y**2` bit for
  bit. The strict `<` comparison picks the lowest index on ties, exactly like
  `argmin`. A different label on a near-tie would change a centroid by far
  more than `1e-4`.
- **Privatised accumulation.** The block adds $x$, $y$ and 1 into
  shared-memory arrays $\Sigma_x, \Sigma_y, \text{cnt}$ (float64 atomics).
  After a barrier, one global atomic per non-empty cluster flushes them. For
  $k = 5$ and $n = 10^4$ this replaces 30 000 heavily contended global
  atomics per iteration with at most $3 \cdot 5 \cdot 40$.

### 2. `updateCentroids` ($k$ Threads)

$\mu_c = \Sigma_c / \text{cnt}_c$ if $\text{cnt}_c > 0$. It then zeroes the
accumulators for the next iteration, which avoids a separate `memset` launch.

## Cost Analysis

$$
W \approx T\,(5nk + 3n), \qquad Q \approx T\,(8n + 4n) \ \text{bytes}
$$

| Symbol | Meaning |
|---|---|
| $W$ | Operations: ~5 per point–centroid distance, plus 3 accumulations per point, per iteration |
| $Q$ | Bytes per iteration: read the coordinates (8 bytes/point) and write the labels (4 bytes/point) |

At the benchmark size, each iteration is ~0.1 MB of traffic and far below
any throughput limit. The cost is **60 kernel launches** (2 × 30). CUDA
Graphs, or a single persistent kernel with a grid barrier, would remove most
of that overhead.

## Pitfalls

- **Accumulating in float32 atomics.** The summation order varies between
  runs, and with $10^6$ points the mean drifts. Float64 shared atomics are
  cheap and stable.
- **Empty clusters** must keep their previous centroid (no division by 0).
- **Labels.** The output labels are those of the *last* assignment, computed
  before the final centroid update, as in the reference.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md), including
$k = 1$, $k > n$ (empty clusters) and duplicated points (distance ties).

## Related

- [Nearest Neighbor](../038-nearest-neighbor/), [Histogramming](../013-histogramming/) (privatisation).

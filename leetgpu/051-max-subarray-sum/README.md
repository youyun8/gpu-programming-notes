---
title: Max Subarray Sum
platform: LeetGPU
upstream: medium/51_max_subarray_sum
url: https://leetgpu.com/challenges/max-subarray-sum
difficulty: medium
tags: [scan, prefix-sum, sliding-window, integer]
status: solved
---

# Max Subarray Sum

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/max-subarray-sum)

## Problem

Find the maximum sum over all contiguous windows of **exactly** $w$ elements
in an int32 array of length $N$ ($N \le 5\times10^4$, values in $[-10, 10]$;
benchmark $N = 5\times10^4$). The result must be exact. The sequential
sliding window is $O(N)$ but serial. The parallel formulation uses **prefix
sums**.

## Formulation

$$
\text{out} = \max_{0 \le i \le N - w}\ \sum_{t = i}^{i + w - 1} x_t
= \max_{0 \le i \le N - w}\ \bigl(P_{i+w} - P_i\bigr), \qquad P_0 = 0,\quad P_{j} = \sum_{t=0}^{j-1} x_t
$$

| Symbol | Meaning |
|---|---|
| $N$ | array length |
| $w$ | window size (`window_size`), $1 \le w \le N$ |
| $x_t$ | input values (int32) |
| $i$ | window start |
| $P_j$ | exclusive prefix sum: sum of the first $j$ elements ($P$ has $N+1$ entries) |
| out | the maximum window sum |

Every window sum becomes **one subtraction** of two prefix values. After
the scan, the problem is a data-parallel max-reduction over $N - w + 1$
independent differences.

## Approach

$N \le 50\,000$ fits comfortably in **one block of 1024 threads**, so no
inter-block scan machinery is needed.

1. **Scan in chunks of 1024.** Each thread loads one element. A
   block-inclusive scan (warp `__shfl_up_sync` scan, a scan of the 32 warp
   totals, then the warp offsets added back) plus a running `carry` from
   previous chunks gives $P_{i+1}$, written to a global `prefix` array. The
   last thread publishes the new carry, with barriers around it.
2. **Max over windows.** Thread $t$ scans $i = t, t + 1024, \dots$, computing
   $P_{i+w} - P_i$ and keeping the max. A warp `__shfl_xor_sync` max and a
   shared-memory pass over the 32 warp maxima finish the job.

## Cost analysis

$$
W_{\text{naive}} = w\,(N - w + 1), \qquad W_{\text{scan}} = O(N), \qquad Q \approx 4N + 4(N+1) + 8(N-w+1) \ \text{bytes}
$$

| Symbol | Meaning |
|---|---|
| $W_{\text{naive}}$ | additions if each window is summed from scratch (up to $6.25\times10^8$ for $w = N/2$) |
| $W_{\text{scan}}$ | work of scan + max |
| $Q$ | bytes: read the input, write the prefix, read two prefix values per window |

Everything fits in L2 (≈ 600 KB). The runtime is a few microseconds of
single-block work plus the launch.

## Pitfalls

- **Off-by-one in the prefix.** Using $P$ with $N + 1$ entries and
  $P_0 = 0$ makes the window $[i, i+w)$ exactly $P_{i+w} - P_i$.
- **All-negative input.** The maximum can be negative, so initialise with
  `INT_MIN`, not 0.
- **Carry race.** Every thread reads `carry` in the scan, so the update must
  sit between two barriers.

## Verification

Exact equality on all LeetGPU cases in [cuemu](../../tools/cuemu/README.md),
including $w = 1$, $w = N$ and all-negative arrays.

## Related

- [Prefix Sum](../016-prefix-sum/), [Subarray Sum](../047-subarray-sum/).

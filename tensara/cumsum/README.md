---
title: Cumulative Sum
platform: Tensara
upstream: cumsum
url: https://tensara.org/problems/cumsum
difficulty: medium
tags: [scan, prefix-sum, fp64-accumulation]
status: solved
---

# Cumulative Sum

**Platform:** Tensara · **Difficulty:** medium · [Problem statement](https://tensara.org/problems/cumsum)

## Problem

Inclusive prefix sum of a float32 vector of length $N$ (64 K … 1 M),
matching `torch.cumsum(x, dim=0)`. The check is `rtol = 3e-2`,
`atol = 1e-2`. This is the textbook scan problem; the solution is written
once, generically, and reused by [cumprod](../cumprod/).

## Formulation

$$
y_i = \sum_{j=0}^{i} x_j
$$

| Symbol | Meaning |
|---|---|
| $x_j$ | Input element, $0 \le j < N$ |
| $y_i$ | Output: the sum of the first $i+1$ inputs |

With chunks of $L$ elements, block $c$ covers $[cL, (c+1)L)$ and

$$
T_c = \sum_{j=cL}^{(c+1)L-1} x_j, \qquad E_c = \sum_{c'=0}^{c-1} T_{c'}, \qquad
y_i = E_c + \sum_{j=cL}^{i} x_j \quad (cL \le i < (c+1)L)
$$

| Symbol | Meaning |
|---|---|
| $L$ | Chunk length, $256 \times 8 = 2048$ |
| $T_c$ | Total of chunk $c$ |
| $E_c$ | Exclusive carry into chunk $c$ |

Inside a block, the warp-level scan uses the Hillis–Steele recurrence

$$
v^{(s+1)}_\ell = \begin{cases} v^{(s)}_{\ell - 2^s} + v^{(s)}_\ell, & \ell \ge 2^s \\ v^{(s)}_\ell, & \text{otherwise} \end{cases}, \qquad s = 0, \dots, 4
$$

| Symbol | Meaning |
|---|---|
| $\ell$ | Lane index, $0 \dots 31$ |
| $v^{(s)}_\ell$ | Lane $\ell$'s partial sum after step $s$ (`__shfl_up_sync` by $2^s$) |

## Approach

1. **`chunkTotals`**: each thread adds its 8 consecutive elements, and a
   block scan of the 256 per-thread values yields $T_c$.
2. **`scanTotals`**: a single block scans the totals (up to 512) in
   256-wide passes, carrying the running total between passes, and writes
   the exclusive carries $E_c$.
3. **`scanChunks`**: each thread reloads its 8 items, the block scans the
   per-thread sums (warp shuffles, then one warp scans the 8 warp totals),
   and each thread walks its 8 items serially starting from
   $E_c + (\text{exclusive prefix of its thread})$.
4. All accumulators are `double`, so rounding does not build up across
   $10^6$ additions; the result is rounded to float once per element.

## Cost Analysis

$$
Q = 12N\ \text{bytes}, \qquad W = 2N\ \text{adds (plus } O(N/8)\text{ in the block scans)}, \qquad T_{\min} = \frac{12N}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes: read twice, write once |
| $W$ | Additions |
| $\beta$ | DRAM bandwidth |

For $N = 2^{20}$ this is ~6 µs of traffic; the three launches and the
middle single-block kernel cost about as much. A single-pass
decoupled look-back scan reads the data once and would reach $8N$ bytes.

## Pitfalls

- **Inclusive**, not exclusive: $y_0 = x_0$.
- **fp32 drift**: with float carries, the relative error grows with $N$;
  the loose tolerance hides it, fp64 removes it.
- **`__syncthreads()` after reading shared totals**: the scan helper is
  called in a loop, so it must not overwrite `warp_totals` while other
  warps still read it.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Cumprod](../cumprod/), [Running Sum 1D](../running-sum-1d/),
  LeetGPU [Prefix Sum](../../leetgpu/016-prefix-sum/),
  LeetGPU [Segmented Prefix Sum](../../leetgpu/070-segmented-prefix-sum/).

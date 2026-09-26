---
title: Array Sorting
platform: Tensara
upstream: array-sort
url: https://tensara.org/problems/array-sort
difficulty: easy
tags: [sorting, radix-sort, bit-tricks]
status: solved
---

# Array Sorting

**Platform:** Tensara · **Difficulty:** easy · [Problem statement](https://tensara.org/problems/array-sort)

## Problem

Sort $n$ **signed** int32 values ascending ($n = 16\,384 \dots 262\,144$),
with an exact check. The GPU sort of choice for fixed-width keys is LSD
radix sort, which works on *unsigned* digits. The only twist is a one-bit
transform that makes signed order equal unsigned order.

## Formulation

$$
f(x) = \operatorname{bits}(x) \oplus \texttt{0x80000000}, \qquad x < y \iff f(x) <_{\text{unsigned}} f(y)
$$

| Symbol | Meaning |
|---|---|
| $x, y$ | signed 32-bit integers (two's complement) |
| $\operatorname{bits}(x)$ | the raw 32-bit pattern |
| $\oplus$ | XOR; flipping the sign bit |
| $f$ | order-preserving map from int32 to uint32 |

Two's complement orders negatives $\texttt{0x80000000} \dots \texttt{0xFFFFFFFF}$
*after* non-negatives when read as unsigned. Flipping the top bit shifts
every value by $2^{31}$, i.e. maps $[-2^{31}, 2^{31})$ monotonically onto
$[0, 2^{32})$.

After mapping, four stable counting-sort passes of 8 bits sort the keys
(see LeetGPU [Radix Sort](../../leetgpu/036-radix-sort/) for the
per-pass scatter formula).

## Approach

1. Map $a \to f(a)$ into the output buffer (as uint32).
2. **Stable LSD radix sort**, 4 passes, ping-ponging with a scratch buffer:
   tile histograms (digit-major), a device-wide exclusive scan, and a stable
   scatter using `__match_any_sync` ranks.
3. Map back with the same XOR (it is its own inverse).

## Cost Analysis

$$
Q \approx 4\cdot 12n + 8n = 56n\ \text{bytes}, \qquad W = O(4n)
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes: per pass, read to count plus read and scatter; plus the two maps |
| $W$ | work, linear in $n$ |

At $n = 262\,144$ that is 15 MB, which is L2-resident. The runtime is
dominated by the ~20 kernel launches. For arrays this small, a single-block
sort (e.g. bitonic in shared memory, or a one-block radix sort) would cut the
launch count.

## Pitfalls

- **Sorting the raw bits** as unsigned puts negatives last.
- **Stability** is essential for LSD correctness.

## Verification

Exact equality on all test cases in [cuemu](../../tools/cuemu/README.md),
including all-negative, all-equal and extreme values
($\pm 2^{31}$ boundaries).

## Related

- LeetGPU [Radix Sort](../../leetgpu/036-radix-sort/), [Sorting (floats)](../../leetgpu/015-sorting/),
  [Top-K](../../leetgpu/029-top-k-selection/).

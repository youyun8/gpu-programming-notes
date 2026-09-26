---
title: Top K Selection
platform: LeetGPU
upstream: medium/29_top_k_selection
url: https://leetgpu.com/challenges/top-k-selection
difficulty: medium
tags: [selection, radix-select, bitonic-sort, bit-tricks]
status: solved
---

# Top K Selection

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/top-k-selection)

## Problem

Return the $k$ largest of $N$ float32 values in descending order
($1 \le k \le N \le 10^8$; benchmark $N = 5\times10^7$, $k = 100$; the
tolerance is exact apart from `atol = 1e-5`). Sorting the whole array would
be $O(N\log N)$ or 4+ radix passes of scatter traffic, even though only 100
values are needed. **Radix select** finds the $k$-th largest value with a few
read-only histogram passes. Only the survivors are then sorted.

## Formulation

Let $x_{(0)} \ge x_{(1)} \ge \dots \ge x_{(N-1)}$ be the input sorted in
descending order. The output is

$$
\text{out}_r = x_{(r)}, \qquad 0 \le r < k
$$

and the **threshold** $\tau = x_{(k-1)}$ satisfies

$$
\#\{\, i : x_i > \tau \,\} \;<\; k \;\le\; \#\{\, i : x_i \ge \tau \,\}
$$

| Symbol | Meaning |
|---|---|
| $N$ | number of input values |
| $k$ | number of values to return |
| $x_{(r)}$ | the $r$-th largest value (order statistic, 0-based) |
| $\tau$ | the $k$-th largest value (threshold) |
| $\#\{\cdot\}$ | number of indices satisfying the condition |

The answer is every element $> \tau$, plus exactly
$k - \#\{x_i > \tau\}$ copies of $\tau$.

### Radix Select on Order-Preserving Keys

Map floats to unsigned keys with $f$ (see [Sorting](../015-sorting/)), so that
key order equals float order. Determine the key $T = f(\tau)$ one 8-bit digit
at a time, from the most significant digit down. Pass $p$ ($p = 3, 2, 1, 0$,
bit shift $8p$) only considers the candidates whose higher digits match the
prefix found so far:

$$
c_d = \#\{\, i : \operatorname{prefix}(f(x_i)) = \Pi,\ \operatorname{digit}_p(f(x_i)) = d \,\}, \qquad
d^\star = \max\Bigl\{ d : \sum_{d' \ge d} c_{d'} \ge \rho \Bigr\}, \qquad
\rho \leftarrow \rho - \sum_{d' > d^\star} c_{d'}
$$

| Symbol | Meaning |
|---|---|
| $\Pi$ | digits of $T$ determined in earlier passes (`prefix`, with `mask` marking which bits are valid) |
| $\operatorname{digit}_p(u)$ | bits $8p \dots 8p+7$ of key $u$ |
| $c_d$ | histogram: candidates whose current digit equals $d$ |
| $\rho$ | rank of $T$ among the current candidates (starts at $k$) |
| $d^\star$ | the digit of $T$ in this pass: walking from $d = 255$ down, the first bucket where the cumulative count reaches $\rho$ |

After 4 passes, $T$ is fully known, and $\rho$ is the number of copies of $T$
that belong to the answer.

## Approach

All state (`prefix`, `mask`, `remaining`, gather cursor, histogram) lives in
`__device__` variables. The sequence of kernels is launched without any
host–device copies:

1. **`digitHistogram` × 4.** A grid-stride pass computes `floatToKey`, and
   candidates matching the prefix increment a shared-memory histogram. Each
   block flushes its non-zero bins with global atomics (privatisation, as in
   [Histogramming](../013-histogramming/)).
2. **`chooseDigit` × 4** (1 thread): the 256-bucket walk above. It also
   resets the histogram for the next pass.
3. **`gatherGreater`.** Every element with key $> T$ is appended to a small
   buffer through an atomic cursor. There are fewer than $k$ such elements.
4. **`fillTail`.** Slots $[\text{count}, k)$ receive $T$, and slots
   $[k, \text{padded})$ receive key 0 (the smallest key, which sorts last).
5. **Bitonic sort, descending**, of the padded power-of-two buffer. It runs
   in one block in shared memory when $k \le 2048$ (the benchmark has
   $k = 100 \to 128$), and otherwise with one global kernel per
   (size, stride) step.
6. **`writeOutput`**: apply $f^{-1}$ to the first $k$ keys.

### Bitonic Sort in One Line

For block size $s = 2, 4, \dots, P$ and stride $t = s/2, \dots, 1$, element
$i$ is compare-exchanged with $j = i \oplus t$. The direction is descending
when $(i \mathbin{\&} s) = 0$. That is $\frac{\log_2 P(\log_2 P + 1)}{2}$
data-independent stages, which suits SIMT perfectly.

## Cost Analysis

$$
Q \approx \underbrace{4 \cdot 4N}_{\text{4 histogram passes}} + \underbrace{4N}_{\text{gather}} = 20N \ \text{bytes}, \qquad
W_{\text{sort}} = O\!\left(P \log^2 P\right),\ P = 2^{\lceil\log_2 k\rceil}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM bytes; every pass only reads the input (4 bytes/element) |
| $P$ | padded sort size (next power of two $\ge k$) |
| $W_{\text{sort}}$ | compare-exchanges in the bitonic sort of the survivors |

Benchmark: $Q = 1$ GB, i.e. ≈ 0.5 ms at 2 TB/s. This is independent of $k$
and about 2–3× less traffic than a full radix sort, which reads **and
scatters** $N$ keys per pass. The bitonic sort of 128 keys is negligible.

## Pitfalls

- **Ties at the threshold.** Using `>=` $\tau$ in the gather can return more
  than $k$ elements. Counting exactly $\rho$ copies of $T$ is what makes
  duplicates correct.
- **Negative floats.** Without the order-preserving key map, negatives rank
  above positives.
- **Padding value.** The padding must sort *after* every real key in
  descending order, which key 0 guarantees (no float maps to 0 except NaN
  patterns, which do not occur in the tests).

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md). Extra tests
cover $k = N$, $k = 1$, all-equal inputs, many duplicates of $\tau$, negative
values, and $k > 2048$ (the global bitonic path).

## Related

- [Sorting](../015-sorting/), [Radix Sort](../036-radix-sort/), [Top-p Sampling](../060-top-p-sampling/),
  [MoE Top-k Gating](../067-moe-topk-gating/).
- Tensara [Argmax](../../tensara/argmax/).

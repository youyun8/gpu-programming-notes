---
title: Sorting
platform: LeetGPU
upstream: hard/15_sorting
url: https://leetgpu.com/challenges/sorting
difficulty: hard
tags: [sorting, radix-sort, bit-tricks, scan]
status: solved
---

# Sorting

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/sorting)

## Problem

Sort $N$ float32 values ascending, in place ($1 \le N \le 10^6$; benchmark
$N = 10^6$). The reference is `torch.sort`. Any algorithm is allowed. On
GPUs the fastest general-purpose choice for 32-bit keys is **LSD radix
sort**, which needs no comparisons at all. Its only obstacle, the sign and
exponent encoding of IEEE floats, is removed with a bit trick.

## Formulation

Find a permutation $\pi$ such that

$$
y_k = x_{\pi(k)}, \qquad y_0 \le y_1 \le \dots \le y_{N-1}
$$

| Symbol | Meaning |
|---|---|
| $N$ | number of elements |
| $x_i$ | input values (float32) |
| $\pi$ | a permutation of $\{0, \dots, N-1\}$ |
| $y_k$ | sorted output, written back to `data` |

### Order-preserving float → integer map

IEEE-754 floats compare like sign-magnitude integers. The map

$$
f(u) =
\begin{cases}
u \oplus \texttt{0xFFFFFFFF}, & \text{sign bit of } u = 1 \ (\text{negative}) \\
u \mathbin{\vert} \texttt{0x80000000}, & \text{sign bit of } u = 0 \ (\text{non-negative})
\end{cases}
$$

is a bijection on 32-bit patterns with
$a < b \iff f(\text{bits}(a)) < f(\text{bits}(b))$ for all non-NaN floats.
Negatives have all bits flipped, so larger magnitudes become smaller
unsigned numbers. Non-negatives only get the top bit set, which places them
above every negative.

| Symbol | Meaning |
|---|---|
| $u$ | raw 32-bit pattern of a float (`__float_as_uint`) |
| $\oplus$ | bitwise XOR |
| $\vert$ | bitwise OR |
| $f(u)$ | unsigned key whose integer order equals the float order |

### LSD radix sort

Write each key in base $2^8$ as digits $(d_3, d_2, d_1, d_0)$. Four
**stable** counting-sort passes on $d_0$, then $d_1$, $d_2$, $d_3$ sort the
keys completely. Stability guarantees that ties on the current digit keep
the order established by the lower digits. In each pass, key $x$ with digit
$d$ in tile $t$ goes to

$$
\text{pos}(x) = \underbrace{\sum_{d' < d}\ \sum_{t'} c_{d',t'} + \sum_{t' < t} c_{d,t'}}_{\text{exclusive scan of } c \text{ in digit-major order}} \;+\; \text{rank of } x \text{ among digit-}d\text{ keys of tile } t
$$

| Symbol | Meaning |
|---|---|
| $d$ | the key's current 8-bit digit, $0..255$ |
| $t$ | tile (block of 2048 keys) containing the key |
| $c_{d,t}$ | number of keys with digit $d$ in tile $t$ |
| rank | number of keys in the same tile with the same digit that come earlier (stability) |

## Approach

All of the radix machinery is shared with [Radix Sort](../036-radix-sort/):

1. `floatToKey`: apply $f$ into a temporary key buffer.
2. For each of the 4 digits (shift 0, 8, 16, 24):
   1. `digitCounts`: 256-bin shared-memory histogram per 2048-key tile,
      stored digit-major `hist[d * tiles + t]`.
   2. `exclusiveScan` of the $256 \times$ tiles table (reduce-then-scan).
   3. `scatterStable`: rank each key within its tile. Inside a warp,
      `__match_any_sync(digit)` returns the mask of lanes with the same digit,
      and `__popc(peers & lanes_below)` is the key's rank among them.
      Per-warp digit counts are prefix-summed across the 8 warps, and a
      running per-digit base carries across the tile's 8 sub-chunks. Every key
      gets a unique, order-preserving slot.
3. After 4 ping-pong passes the keys are back in the original buffer.
   `keyToFloat` applies $f^{-1}$ and writes into `data`.

## Cost analysis

$$
Q \approx \underbrace{8N}_{\text{map in/out}} + P\,(\underbrace{4N}_{\text{count}} + \underbrace{8N}_{\text{scatter}}) + 8N, \qquad P = \frac{32}{8} = 4
$$

| Symbol | Meaning |
|---|---|
| $Q$ | DRAM traffic in bytes (the histogram tables are negligible) |
| $P$ | number of radix passes (32-bit keys, 8-bit digits) |
| $4N$ | reading keys to count digits |
| $8N$ | reading and scattering keys |

For $N = 10^6$, $Q \approx 64$ MB, tens of microseconds of bandwidth. The work
is $O(PN)$ versus $O(N\log^2 N)$ for a bitonic sort. With about 20 kernel
launches, launch overhead is a noticeable share of the runtime at this size.

## Pitfalls

- **Negative floats.** Sorting raw bit patterns as unsigned integers puts
  negatives after positives, and in reverse order. The map $f$ fixes both.
- **Stability.** A scatter with `atomicAdd` on per-digit counters is fast but
  unstable, which breaks LSD radix sort. The match-any ranking keeps it
  stable.
- **$-0.0$ vs $+0.0$.** They map to adjacent keys ($-0 < +0$). They compare
  equal as floats, so either order is accepted.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md), and extra
tests cover all-equal, reversed, $\pm 0$, denormals and $\pm\infty$ inputs.

## Related

- [Radix Sort](../036-radix-sort/) (unsigned integers), [Top-K](../029-top-k-selection/),
  [Prefix Sum](../016-prefix-sum/) (the scan used here).
- Tensara [Array Sort](../../tensara/array-sort/).

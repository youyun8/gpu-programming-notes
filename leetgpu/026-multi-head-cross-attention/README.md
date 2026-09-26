---
title: Multi-Head Cross-Attention
platform: LeetGPU
upstream: hard/26_multi_head_cross_attention
url: https://leetgpu.com/challenges/multi-head-cross-attention
difficulty: hard
tags: [attention, cross-attention, flash-attention, multi-head]
status: solved
---

# Multi-Head Cross-Attention

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/multi-head-cross-attention)

## Problem

Multi-head **cross**-attention as in encoder–decoder transformers (T5,
Whisper, Stable Diffusion's text conditioning). Decoder queries $Q$ have
shape $(M, H, D)$, and encoder keys/values $K, V$ have shape $(N, H, D)$. The
output is $(M, H, D)$ ($M, N \le 4096$, $H \le 64$, $D \le 256$; benchmark
$M = 1024$, $N = 2048$, $H = 16$, $D = 128$; tolerance `1e-4`). There is no
mask, and $M \ne N$ in general.

## Formulation

For each head $h$:

$$
O_{i,h,:} = \sum_{j=0}^{N-1} \frac{e^{s_{ij}^{(h)} - m_i^{(h)}}}{\sum_{j'} e^{s_{ij'}^{(h)} - m_i^{(h)}}}\, V_{j,h,:}, \qquad
s_{ij}^{(h)} = \frac{1}{\sqrt D}\sum_{c=0}^{D-1} Q_{i,h,c}\, K_{j,h,c}
$$

| Symbol | Meaning |
|---|---|
| $M$ | number of decoder queries |
| $N$ | number of encoder positions (keys/values) |
| $H$ | number of heads |
| $D$ | head dimension |
| $Q_{i,h,c}$ | query $i$, head $h$, feature $c$; offset $(iH + h)D + c$ |
| $K_{j,h,c},\ V_{j,h,c}$ | key/value $j$, head $h$, feature $c$; offset $(jH + h)D + c$ |
| $s^{(h)}_{ij}$ | scaled score of query $i$ vs key $j$ in head $h$ |
| $m^{(h)}_i$ | row maximum $\max_j s^{(h)}_{ij}$ |
| $O_{i,h,:}$ | output vector (length $D$) at offset $(iH + h)D$ |

### The transposes are free

The reference transposes $(M, H, D) \to (H, M, D)$ before a batched matmul.
In memory, row $i$ of head $h$ starts at $(iH + h)D$. Head $h$ is therefore a
matrix with **base offset $hD$** and **row stride $HD$**. The strided flash
kernel from [Multi-Head Attention](../012-multi-head-attention/) takes exactly
those parameters, so no data is transposed or copied:

| `AttnGeom` field | Value |
|---|---|
| `q_rows`, `kv_rows` | $M$, $N$ |
| `head_dim` | $D$ |
| `q_stride`, `kv_stride`, `o_stride` | $H \cdot D$ |
| `q_head`, `kv_head`, `o_head` | $D$ |
| `scale` | $1/\sqrt D$ |

## Approach

The kernel is the generic FlashAttention-style forward:

- grid $\lceil M/4\rceil \times H$, 4 warps per block, one query row per warp;
- 32-key tiles of $K$/$V$ streamed through shared memory in 128-wide slices
  of $D$;
- lane $\ell$ scores key $\ell$; online softmax with correction
  $\alpha = e^{m - m'}$; $PV$ accumulation with `__shfl_sync` broadcasts of
  $p_j$; register accumulators (up to 8 columns per lane for $D = 256$).

See [Softmax Attention](../006-softmax-attention/) for the per-tile algebra
and [Multi-Head Attention](../012-multi-head-attention/) for the head-slicing
details.

## Cost analysis

$$
W = 4MNHD, \qquad Q_{\min} = 4\,(2MHD + 2NHD), \qquad I_{\max} = \frac{W}{Q_{\min}} = \frac{MN}{2(M+N)}
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs: $QK^{\mathsf T}$ and $PV$, $2MND$ each, per head |
| $Q_{\min}$ | compulsory bytes: read $Q$, $K$, $V$ and write the output once each |
| $I_{\max}$ | best achievable arithmetic intensity (FLOP/byte) |

Benchmark: $W \approx 17$ GFLOP, $Q_{\min} \approx 50$ MB, and
$I_{\max} \approx 340$, so it is solidly **compute-bound**. With 4 query rows
per block, each K/V tile is reused only 4×. The kernel is limited by fp32
FMA and shared-load throughput, not DRAM. A tensor-core version with
64–128-query tiles would be the next step.

## Pitfalls

- **Scale.** $1/\sqrt D$ (the head dimension), not $1/\sqrt{HD}$.
- **$M \ne N$.** `q_rows` and `kv_rows` are separate, and row guards use
  `q_rows`.
- **Shared memory at $D = 256$.** $4D + 32\cdot129 + 32\cdot128$ floats
  ≈ 36 KB, under the 48 KB default. The attribute call keeps larger $D$ safe.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-4`,
including $M = 1$, $N = 1$ and $D = 256$.

## Related

- [Multi-Head Attention](../012-multi-head-attention/), [Grouped-Query Attention](../080-grouped-query-attention/),
  [Softmax Attention](../006-softmax-attention/).

---
title: Softmax Attention
platform: LeetGPU
upstream: medium/6_softmax_attention
url: https://leetgpu.com/challenges/softmax-attention
difficulty: medium
tags: [attention, flash-attention, online-softmax, shared-memory, warp-shuffle]
status: solved
---

# Softmax Attention

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/softmax-attention)

## Problem

Compute single-head scaled dot-product attention in float32:
$Q \in \mathbb{R}^{M\times d}$, $K, V \in \mathbb{R}^{N \times d}$,
output $\in \mathbb{R}^{M\times d}$ ($1 \le M, N \le 10^5$, $1 \le d \le 128$;
benchmark $M = 512$, $N = 256$). The tolerance is `1e-4`. With $M = N = 10^5$
the score matrix alone would take 40 GB, so it must **never be materialised**.
This is the core idea of FlashAttention.

## Formulation

$$
\text{Attention}(Q, K, V) = \operatorname{softmax}_{\text{row}}\!\left(\frac{QK^{\mathsf T}}{\sqrt d}\right) V
$$

Written per query row $r$:

$$
s_{rj} = \frac{1}{\sqrt d}\sum_{c=0}^{d-1} Q_{rc} K_{jc}, \qquad
p_{rj} = \frac{e^{s_{rj} - m_r}}{\sum_{j'} e^{s_{rj'} - m_r}}, \qquad
O_{rc} = \sum_{j=0}^{N-1} p_{rj} V_{jc}
$$

| Symbol | Meaning |
|---|---|
| $M$ | Number of queries (rows of $Q$ and of the output) |
| $N$ | Number of keys/values (rows of $K$, $V$) |
| $d$ | Head dimension (columns of $Q$, $K$, $V$, output), $\le 128$ |
| $r,\ j,\ c$ | Query index, key index, feature (column) index |
| $s_{rj}$ | Scaled attention score of query $r$ against key $j$ |
| $m_r$ | $\max_j s_{rj}$, for numerical stability |
| $p_{rj}$ | Attention weight; each row sums to 1 |
| $O_{rc}$ | Output element (`output[r*d + c]`) |

### Online Softmax over Key Tiles

Process the keys in tiles $\mathcal{T}_1, \mathcal{T}_2, \dots$ of 32 keys.
After each tile, update the running max $m$, denominator $\ell$, and
unnormalised output vector $\mathbf{a} \in \mathbb{R}^d$:

$$
m' = \max\!\Bigl(m,\ \max_{j\in\mathcal T} s_{rj}\Bigr), \quad
\alpha = e^{m - m'}, \quad
\ell' = \alpha\,\ell + \sum_{j\in\mathcal T} e^{s_{rj}-m'}, \quad
\mathbf a' = \alpha\,\mathbf a + \sum_{j\in\mathcal T} e^{s_{rj}-m'}\, V_{j,:}
$$

and at the end $O_{r,:} = \mathbf a / \ell$.

| Symbol | Meaning |
|---|---|
| $\mathcal T$ | Current tile of key indices (32 consecutive keys) |
| $m,\ m'$ | Running maximum score before and after the tile (starts at $-\infty$) |
| $\alpha$ | Correction factor that rescales everything accumulated so far to the new maximum |
| $\ell,\ \ell'$ | Running softmax denominator (starts at 0) |
| $\mathbf a,\ \mathbf a'$ | Running un-normalised output row, length $d$ (starts at 0) |
| $V_{j,:}$ | Row $j$ of $V$ |

This is the pair merge from [Softmax](../005-softmax/), extended with a
vector-valued payload $\mathbf a$ that is rescaled by the same $\alpha$.

## Approach

### Work Mapping

| Level | Responsibility |
|---|---|
| Block (128 threads = 4 warps) | 4 consecutive query rows; stages each K/V tile once for all 4 |
| Warp | One query row $r$ |
| Lane $\ell$ | Scores key $\ell$ of the tile; owns output columns $\ell, \ell+32, \ell+64, \ell+96$ |

### Per Tile

1. **Stage.** `__syncthreads()`, then the block copies 32 rows of $K$ and $V$
   into shared memory (`k_tile`, `v_tile`), zero-filling past $N$. A second
   `__syncthreads()` follows.
2. **Score: one key per lane.** Lane $\ell$ computes the full $d$-length dot
   product of the query (pre-scaled by $1/\sqrt d$, kept in shared memory)
   with key $\ell$. A whole tile of scores then costs one `warpMax` and one
   `warpSum` (5 shuffles each), instead of one 5-shuffle reduction *per key*.
3. **Rescale.** Apply the formulas above: $\alpha$ multiplies $\ell$ and the
   4 accumulator registers.
4. **Accumulate $PV$.** For each key $j$ of the tile, broadcast $p_j$ from
   lane $j$ with `__shfl_sync(…, p, j)`. Every lane then does
   $a_c \mathrel{+}= p_j V_{jc}$ for its 4 columns.

### Bank-Conflict Avoidance

In step 2, lane $\ell$ reads `k_tile[ℓ][c]` for the same $c$ as all other
lanes, i.e. a column of the tile. With a row pitch of 128 floats, all lanes
would hit bank $c \bmod 32$. The pitch is therefore **129** (odd), which
spreads the 32 lanes over 32 banks.

## Cost Analysis

$$
W = 4MNd, \qquad
Q_{\text{DRAM}} \approx 4\left(Md + \left\lceil\frac{M}{4}\right\rceil\cdot 2Nd + Md\right), \qquad
\text{mem}_{\text{scores}} = 0
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs: $2MNd$ for $QK^{\mathsf T}$ plus $2MNd$ for $PV$ |
| $Q_{\text{DRAM}}$ | Bytes: read $Q$, every block re-reads all of $K$ and $V$, write the output |
| $\lceil M/4 \rceil$ | Number of blocks (4 query rows each) |
| $\text{mem}_{\text{scores}}$ | Extra memory for the $M \times N$ score matrix, which is never stored |

At the benchmark size ($M = 512$, $N = 256$, $d \le 128$),
$W \approx 67$ MFLOP, and $K$/$V$ (256 KB) stay in L2. Sharing each tile
across 4 rows cuts shared-memory staging traffic by 4×. Larger row blocks
(e.g. 64 queries on tensor cores, as in FlashAttention-2) raise the reuse
further; see [Multi-Head Attention](../012-multi-head-attention/).

## Pitfalls

- **Barriers in inactive warps.** Warps whose row is $\ge M$ must still
  execute every `__syncthreads()` of the tile loop. They skip only the final
  store; returning early would deadlock the block.
- **Partial last tile.** Lanes beyond $N$ get score $-\infty$ and $p = 0$, and
  the $PV$ loop runs only over the `tile_keys` valid keys.
- **Scaling order.** Pre-multiply $Q$ by $1/\sqrt d$ once, instead of every
  score.
- **Initial state.** $m = -\text{FLT\_MAX}$ and $\ell = 0$. The first tile's
  $\alpha = e^{-\text{FLT\_MAX} - m'}$ underflows cleanly to 0.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md), including
$d = 1$, $d = 128$, $N < 32$ and $M$ not a multiple of 4. The runs use
`--reverse` scheduling to check that the staging barriers are sufficient.

## Related

- [Multi-Head Attention](../012-multi-head-attention/), [Causal Attention](../053-casual-attention/),
  [Sliding Window Attention](../059-sliding-window-attn/), [GQA](../080-grouped-query-attention/).
- Tensara [Scaled Dot-Product Attention](../../tensara/scaled-dot-attention/).

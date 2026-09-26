---
title: INT8 KV-Cache Attention
platform: LeetGPU
upstream: medium/96_int8_kv_cache_attention
url: https://leetgpu.com/challenges/int8-kv-cache-attention
difficulty: medium
tags: [attention, decode, flash-decoding, int8, split-k]
status: solved
---

# INT8 KV-Cache Attention

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/int8-kv-cache-attention)

## Problem

**Decode-phase** multi-head attention: one new query token per head against a
long KV cache stored as **int8** with per-token scales
($H \le 64$ heads, $S \le 32\,768$ cached tokens, head dim $8 \le D \le 256$;
benchmark $H = 32$, $S = 8192$, $D = 128$; tolerance `1e-3`). int8 caches
halve memory traffic compared with fp16 (and quarter it compared with fp32).
Decode attention is purely bandwidth-bound, so that translates directly
into speed.

## Formulation

$$
K_{h,j,c} = \kappa_{h,j}\,\hat K_{h,j,c}, \qquad V_{h,j,c} = \nu_{h,j}\,\hat V_{h,j,c}
$$

$$
s_{h,j} = \frac{\mathbf q_h\cdot K_{h,j,:}}{\sqrt D} = \frac{\kappa_{h,j}}{\sqrt D}\sum_c q_{h,c}\,\hat K_{h,j,c}, \qquad
\mathbf o_h = \sum_{j} \frac{e^{s_{h,j} - m_h}}{\sum_{j'} e^{s_{h,j'} - m_h}}\,\nu_{h,j}\,\hat V_{h,j,:}
$$

| Symbol | Meaning |
|---|---|
| $H,\ S,\ D$ | heads, cached sequence length, head dimension |
| $\hat K,\ \hat V$ | int8 cache values in $[-128, 127]$ |
| $\kappa_{h,j},\ \nu_{h,j}$ | per-token float scales (`k_scale`, `v_scale`) |
| $\mathbf q_h$ | the query of head $h$ (length $D$) |
| $s_{h,j}$ | attention score of cached token $j$ |
| $m_h$ | maximum score of head $h$ |
| $\mathbf o_h$ | output of head $h$ |

The per-token scale factors out of the inner sums, so the dot product can
run on the raw int8 values and be scaled once per token.

### Flash-Decoding (Split Along the Sequence)

With one query per head there are only $H$ independent problems, e.g. 32
blocks for 108–132 SMs. Split the keys into chunks $\mathcal C_1, \mathcal C_2, \dots$
of 256 tokens, compute a partial softmax per chunk, and merge:

$$
m = \max_i m_i, \qquad \ell = \sum_i \ell_i\,e^{m_i - m}, \qquad \mathbf o = \frac{1}{\ell}\sum_i e^{m_i - m}\,\mathbf a_i
$$

| Symbol | Meaning |
|---|---|
| $m_i,\ \ell_i$ | max score and sum of $e^{s - m_i}$ over chunk $i$ |
| $\mathbf a_i$ | un-normalised partial output $\sum_{j\in\mathcal C_i} e^{s_j - m_i}V_j$ |
| $m,\ \ell,\ \mathbf o$ | merged statistics and final output |

This is the same associative $(m, \ell, \mathbf a)$ merge as in
[Softmax Attention](../006-softmax-attention/), applied across blocks instead
of across tiles.

## Approach

1. **`partialAttention`**, grid $(\lceil S/256\rceil, H)$ = 1024 blocks at the
   benchmark:
   - **scores**: warp per key, lanes over $D$, int8 loads converted to float,
     a shuffle reduction, then multiplied by $\kappa_j/\sqrt D$ and stored in
     shared memory;
   - **local softmax**: block max, exponentiate, block sum;
   - **$PV$**: thread $c$ (over $D$) accumulates $\sum_j p_j\,\nu_j\hat V_{j,c}$.
     For each key, the $D$ threads read consecutive int8 bytes, which is
     coalesced.
   - It writes $(m_i, \ell_i, \mathbf a_i)$ to scratch.
2. **`combine`**: one block per head merges the chunk partials with the
   formulas above.

## Cost Analysis

$$
Q \approx 2HSD + 8HS + 4HD\cdot 2 + \text{partials}, \qquad W \approx 4HSD
$$

| Symbol | Meaning |
|---|---|
| $Q$ | bytes: int8 K and V (1 byte each), two float scales per token, query and output |
| $W$ | FLOPs (scores and the weighted sum) |

Benchmark: $Q \approx 67$ MB versus 268 MB for a float32 cache, i.e. about
34 µs at 2 TB/s. $W/Q \approx 2$ FLOP/byte, so the kernel is firmly
bandwidth-bound, and the int8 cache is a real 4× win over fp32.

## Pitfalls

- **Too few blocks** without the split (one block per head), leaving most of
  the SMs idle.
- **Scales per token**: apply $\kappa$ to the whole dot product and $\nu$ to
  the value row, not per element twice.
- **Empty chunks** do not exist ($S \ge 1$), but the last chunk is partial.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-3`,
including $S = 1$ and $S$ not divisible by 256.

## Related

- [GQA](../080-grouped-query-attention/), [Softmax Attention](../006-softmax-attention/),
  [INT8 MatMul](../032-int8-quantized-matmul/).

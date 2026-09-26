---
title: Multi-Head Attention
platform: LeetGPU
upstream: hard/12_multi_head_attention
url: https://leetgpu.com/challenges/multi-head-attention
difficulty: hard
tags: [attention, flash-attention, multi-head, online-softmax]
status: solved
---

# Multi-Head Attention

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/multi-head-attention)

## Problem

Multi-head self-attention without projections. $Q, K, V$ are $N \times d_{\text{model}}$
float32 matrices, split column-wise into $h$ heads of width $d_k = d_{\text{model}}/h$
($1 \le N \le 10^4$, $2 \le d_{\text{model}} \le 1024$, $1 \le h \le d_{\text{model}}$;
benchmark $N = d_{\text{model}} = 1024$; tolerance `1e-5`). The heads are
concatenated back into an $N \times d_{\text{model}}$ output. The hard part is
the range of $d_k$, from 1 (with $h = d_{\text{model}}$) to 1024 (with
$h = 1$), which rules out a design with a fixed register or shared-memory
budget per row.

## Formulation

$$
\operatorname{MultiHead}(Q, K, V) = \operatorname{Concat}(H_0, \dots, H_{h-1}), \qquad
H_i = \operatorname{softmax}_{\text{row}}\!\left(\frac{Q_i K_i^{\mathsf T}}{\sqrt{d_k}}\right) V_i
$$

$$
Q_i = Q[:,\ i d_k : (i+1) d_k], \quad K_i = K[:,\ i d_k : (i+1) d_k], \quad V_i = V[:,\ i d_k : (i+1) d_k]
$$

| Symbol | Meaning |
|---|---|
| $N$ | sequence length (rows of $Q$, $K$, $V$, output) |
| $d_{\text{model}}$ | model width (columns) |
| $h$ | number of heads |
| $d_k$ | head width $d_{\text{model}}/h$ |
| $Q_i,\ K_i,\ V_i$ | column block $i$ of the inputs, each $N \times d_k$ |
| $H_i$ | output of head $i$ ($N \times d_k$), written to columns $[i d_k, (i+1)d_k)$ |
| $\operatorname{softmax}_{\text{row}}$ | softmax applied independently to each row |

The per-row online-softmax update is the one derived in
[Softmax Attention](../006-softmax-attention/): running max $m$, denominator
$\ell$, accumulator $\mathbf a$, and correction $\alpha = e^{m - m'}$.

### Heads as Strides

Element $(r, c)$ of head $i$ lives at offset
$r \cdot d_{\text{model}} + i\,d_k + c$. A head is therefore just a base
pointer ($+\,i\,d_k$) and a row stride ($d_{\text{model}}$). Splitting and
concatenating heads need **no data movement**. The kernel receives a small
`AttnGeom` struct (rows, head_dim, row strides, per-head offsets, scale), and
the same kernel is reused by the causal, GQA and other attention variants.

## Approach

### Grid

$\lceil N/4\rceil \times h$ blocks of 128 threads. `blockIdx.y` selects the
head, and each warp handles one query row.

### Slicing the Head Dimension

$d_k$ can be as large as 1024, but a $32 \times 1024$ float tile would need
128 KB of shared memory. The kernel therefore streams K and V through shared
memory in **slices of 128 columns**:

1. **Scores.** For each key tile of 32, loop over the slices
   $c_0 = 0, 128, \dots$. Stage the $32 \times 128$ slice of $K$ (pitch 129
   to avoid bank conflicts), and let lane $\ell$ add the partial dot product
   of the query slice with key $\ell$. After the last slice, lane $\ell$
   holds the full score $s_\ell$.
2. **Online softmax.** One `warpMax` and one `warpSum` for the tile, then the
   correction $\alpha$ is applied to the accumulators.
3. **$PV$.** Loop over the slices again: stage the $32 \times 128$ slice of
   $V$, broadcast each $p_j$ with `__shfl_sync`, and update the accumulator
   registers of that slice.

### Keeping Accumulators in Registers

Each lane owns up to $8 \text{ slices} \times 4 = 32$ output columns. The
slice loop is `#pragma unroll`ed with a compile-time bound (`kMaxSlices = 8`)
and breaks early when $c_0 \ge d_k$. Every index into `acc[]` is therefore a
compile-time constant, and the array stays in registers instead of spilling
to local memory.

Shared memory is $4 d_k + 32\cdot129 + 32\cdot128$ floats ≈ 49 KB at
$d_k = 1024$, over the 48 KB default, so the launcher raises the limit with
`cudaFuncSetAttribute(…MaxDynamicSharedMemorySize…)`.

## Cost Analysis

$$
W = 4N^2 d_{\text{model}}, \qquad
Q_{\text{DRAM}} \approx 4\left(2Nd_{\text{model}} + \left\lceil\frac{N}{4}\right\rceil \cdot 2N d_{\text{model}}\right)
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs: over all heads, $2N^2d_k$ for scores plus $2N^2 d_k$ for $PV$, times $h$ |
| $Q_{\text{DRAM}}$ | bytes: read $Q$ and write the output once; each 4-row block streams all keys and values of its head |

At $N = d_{\text{model}} = 1024$, $W \approx 4.3$ GFLOP. $K$ and $V$ (8 MB)
fit in L2, so the $\lceil N/4\rceil$-fold re-reading is served mostly by L2.
The limiting factor is FP32 FMA throughput through shared memory. A
tensor-core version (bf16/tf32 `mma.sync` with 64-query tiles) is the natural
next step.

## Pitfalls

- **Head indexing.** The row stride is $d_{\text{model}}$, not $d_k$. Using
  $d_k$ reads the wrong elements for every head but the first.
- **$d_k < 32$.** Many lanes have no output columns. The `c < width` guards
  keep them idle, but they still take part in shuffles and barriers.
- **Dynamic shared memory above 48 KB** needs the explicit attribute, or the
  launch fails silently.
- **Scale.** $1/\sqrt{d_k}$, not $1/\sqrt{d_{\text{model}}}$.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md), including
$h = d_{\text{model}}$ ($d_k = 1$) and $h = 1$ ($d_k = 1024$). A separate
stress test compares $d_k = 1024$ against PyTorch.

## Related

- [Softmax Attention](../006-softmax-attention/), [Multi-Head Cross-Attention](../026-multi-head-cross-attention/),
  [GQA](../080-grouped-query-attention/), [MLA](../114-multi-head-latent-attention/).

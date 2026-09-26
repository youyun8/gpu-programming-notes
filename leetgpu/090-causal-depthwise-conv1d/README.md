---
title: Causal Depthwise Conv1d
platform: LeetGPU
upstream: medium/90_causal_depthwise_conv1d
url: https://leetgpu.com/challenges/causal-depthwise-conv1d
difficulty: medium
tags: [convolution, depthwise, ssm, channels-last]
status: solved
---

# Causal Depthwise Conv1d

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/causal-depthwise-conv1d)

## Problem

Causal depthwise 1-D convolution over a channels-last tensor
$x \in \mathbb R^{B\times L\times D}$. Each channel $d$ has its own
$K$-tap filter, and output position $l$ only sees inputs $l-K+1 \dots l$
($B \le 16$, $L, D \le 8192$, $K \le 8$; tolerance `1e-4`). This is the
short convolution that Mamba applies before its selective scan to mix local
context within each channel.

## Formulation

$$
y_{b,l,d} = \beta_d + \sum_{k=0}^{K-1} w_{d,k}\; x_{b,\,l-k,\,d}, \qquad x_{b,\,l',\,d} = 0 \ \text{for}\ l' < 0
$$

$$
\text{offset}(b, l, d) = (bL + l)\,D + d \qquad (\text{channels-last})
$$

| Symbol | Meaning |
|---|---|
| $B,\ L,\ D$ | Batch, sequence length, channels |
| $K$ | Kernel taps ($\le 8$) |
| $x_{b,l,d}$ | Input; zero before the start of the sequence (causal left padding) |
| $w_{d,k}$ | Weight of channel $d$ for lag $k$: $w_{d,0}$ applies to the current position, $w_{d,K-1}$ to the oldest |
| $\beta_d$ | per-channel bias |
| $y_{b,l,d}$ | Output, same layout as $x$ |

"Depthwise" means there is no mixing across channels: $D$ independent 1-D
filters. Parameters: $D(K+1)$, versus $D^2K$ for a full convolution.

## Approach

- **Grid** $\lceil D/128\rceil \times \lceil L/8\rceil \times B$ with 128
  threads per block, `threadIdx.x` along **channels**.
- In channels-last layout, a fixed $(b, l)$ is a contiguous row of $D$
  floats, so every tap read `x[b, l-k, :]` by a warp is one coalesced
  128-byte segment.
- Each thread loads its channel's $\le 8$ weights and bias into **registers**
  once, then computes **8 consecutive positions**, reusing them.
- Taps are accumulated oldest-first ($k = K-1 \to 0$), matching `F.conv1d` on
  the left-padded, flipped kernel, so the rounding order is the same.

Each input element is read by up to $K$ different output positions of the
same thread (overlapping windows). Those re-reads hit L1.

## Cost Analysis

$$
W = 2BLDK, \qquad Q_{\min} = 8BLD + 4D(K+1)\ \text{bytes}, \qquad I \approx \frac{K}{4}
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs |
| $Q_{\min}$ | Compulsory bytes: read $x$ once, write $y$ once (plus the negligible weights) |
| $I$ | Arithmetic intensity |

With $K \le 8$, $I \le 2$ FLOP/byte, so the kernel is memory-bound. For
$B = 16$, $L = 8192$, $D = 8192$: $Q = 8.6$ GB, i.e. ≈ 4.3 ms at 2 TB/s.

## Pitfalls

- **Kernel orientation.** `weight[d, 0]` multiplies the **current** input.
  The reference flips the kernel before calling `conv1d`, so a plain
  cross-correlation with `weight[d, k]` at offset $+k$ would be wrong.
- **Causal padding** only on the left, $K-1$ zeros.
- **Layout.** Channels-last means $d$ is fastest. Mapping threads along $l$
  would make every access strided by $D$.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-4`,
including $K = 1$ and $L < K$.

## Related

- [SSM Selective Scan](../094-ssm-selective-scan/), [1D Convolution](../009-1d-convolution/),
  [Linear Recurrence](../082-linear-recurrence/).

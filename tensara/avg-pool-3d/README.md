---
title: 3D Average Pooling
platform: Tensara
upstream: avg-pool-3d
url: https://tensara.org/problems/avg-pool-3d
difficulty: hard
tags: [pooling, stencil, grid-stride]
status: solved
---

# 3D Average Pooling

**Platform:** Tensara · **Difficulty:** hard · [Problem statement](https://tensara.org/problems/avg-pool-3d)

## Problem

3-D average pooling of an $H\times W\times D$ float32 tensor with a
$k\times k\times k$ window, stride $S$ and zero padding $P$, matching
`torch.nn.functional.avg_pool3d` (`count_include_pad=True`, so the divisor is
always $k^3$). The check is `rtol = 2e-4`, `atol = 1e-5`.

## Formulation

$$
X_{\text{out}} = \left\lfloor \frac{X + 2P - k}{S} \right\rfloor + 1 \quad \text{for } X \in \{H, W, D\}
$$

$$
\text{out}[a, b, c] = \frac{1}{k^3} \sum_{m=0}^{k-1}\sum_{n=0}^{k-1}\sum_{o=0}^{k-1}
\tilde{x}\bigl[S a + m - P,\ S b + n - P,\ S c + o - P\bigr]
$$

| Symbol | Meaning |
|---|---|
| $x$ | input tensor $H\times W\times D$, row-major ($D$ is contiguous) |
| $\tilde{x}$ | $x$ extended with zeros outside its bounds |
| $k, S, P$ | window side, stride, padding |
| $X_{\text{out}}$ | output extent along an axis of input extent $X$ |
| $a, b, c$ | output indices along $H, W, D$ |
| $m, n, o$ | offsets inside the window |
| $k^3$ | divisor, always the full window volume |

The output index is decoded from the flat index $t$ as

$$
c = t \bmod D_{\text{out}}, \qquad b = \left\lfloor t / D_{\text{out}} \right\rfloor \bmod W_{\text{out}}, \qquad
a = \left\lfloor t / (D_{\text{out}} W_{\text{out}}) \right\rfloor
$$

| Symbol | Meaning |
|---|---|
| $t$ | flat output index handled by a thread |

## Approach

One thread per output (grid-stride), with three nested window loops that
skip out-of-bounds planes, rows and columns early. Consecutive threads have
consecutive $c$, the contiguous axis, so the innermost loop reads are
coalesced across the warp and overlapping windows are cache hits. The sum
is divided by $k^3$.

## Cost Analysis

$$
W_{\text{ops}} = k^3\,H_{\text{out}} W_{\text{out}} D_{\text{out}}, \qquad
Q \approx 4\,(HWD + H_{\text{out}}W_{\text{out}}D_{\text{out}})\ \text{bytes}
$$

| Symbol | Meaning |
|---|---|
| $W_{\text{ops}}$ | additions |
| $Q$ | compulsory DRAM traffic (each input read once if caches hold the overlap) |

With $S < k$ the windows overlap along three axes, and the working set of a
block ($k$ planes of $k$ rows) is larger than in 2-D. If L2 thrashes, a
separable version (three 1-D passes, $3k$ operations per output instead of
$k^3$) is the next step.

## Pitfalls

- **Divisor $k^3$**, even for windows that stick out of the tensor.
- **The spec overloads $k$** (window size and third output index); the code
  names them differently.
- Index decode order: $D$ is innermost.

## Verification

All test cases (scaled-down variants of the official sizes) pass on
[cuemu](../../tools/cuemu/README.md) against the PyTorch reference.

## Related

- [Avg Pool 1D](../avg-pool-1d/), [Avg Pool 2D](../avg-pool-2d/),
  [Max Pool 3D](../max-pool-3d/), [Conv Square 3D](../conv-square-3d/).

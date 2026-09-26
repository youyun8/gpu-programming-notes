---
title: 1D Convolution
platform: LeetGPU
upstream: easy/9_1d_convolution
url: https://leetgpu.com/challenges/1d-convolution
difficulty: easy
tags: [convolution, shared-memory, dynamic-shared-memory, register-blocking]
status: solved
---

# 1D Convolution

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/1d-convolution)

## Problem

"Valid" 1-D convolution (strictly, cross-correlation, since the kernel is not
flipped) of a float32 signal of length $L$ with a kernel of length $K$
($1 \le K \le 2047$, $K \le L \le 1.5\times10^6$; benchmark $L = 1.5\times10^6$,
$K = 2047$). The output has length $L - K + 1$ and the tolerance is `1e-4`.
With $K$ in the thousands, the naive kernel re-reads every input element $K$
times from global memory. The fix is **shared-memory tiling of the input
window**.

## Formulation

$$
y_i = \sum_{j=0}^{K-1} x_{i+j}\, w_j, \qquad 0 \le i < L - K + 1
$$

| Symbol | Meaning |
|---|---|
| $L$ | input length (`input_size`) |
| $K$ | kernel length (`kernel_size`) |
| $x_n$ | input signal, $0 \le n < L$ |
| $w_j$ | kernel weight, $0 \le j < K$ |
| $y_i$ | output sample; there are $L - K + 1$ of them ("valid" positions only) |

### Tiling

Block $b$ produces the $T = 1024$ outputs $y_{bT}, \dots, y_{bT+T-1}$. They
depend only on the input window

$$
x_{bT},\ \dots,\ x_{bT + T + K - 2} \qquad (\text{length } T + K - 1)
$$

| Symbol | Meaning |
|---|---|
| $b$ | block index |
| $T$ | outputs per block: 256 threads × 4 outputs per thread = 1024 |
| $T + K - 1$ | input window (outputs plus the kernel "halo") staged in shared memory |

## Approach

1. **Stage.** The block copies all $K$ weights and its $T + K - 1$ input
   values into **dynamic** shared memory. The size depends on $K$, so it is
   passed as the third launch parameter:
   $(2K - 1 + T)\cdot 4$ bytes, i.e. 20.4 KB at $K = 2047$. Window elements
   past the end of the input are zero-filled; they only feed outputs that are
   never written.
2. `__syncthreads()`.
3. **Compute.** Thread $t$ owns outputs $t,\ t{+}256,\ t{+}512,\ t{+}768$ of
   the block. For every $j$ it loads $w_j$ once (a **broadcast**: all threads
   read the same word, which costs one transaction) and does 4 FMAs with
   `s_input[t + 256r + j]`. For fixed $j$ and $r$, consecutive threads read
   consecutive words, so there are no bank conflicts.
4. Store the 4 results with a bounds check.

Giving each thread 4 outputs reuses each weight load 4 times from a register
and gives the scheduler 4 independent FMA chains, which hides FMA latency.

## Cost Analysis

$$
W = 2K(L-K+1), \qquad
Q_{\text{naive}} \approx 4\cdot 2K(L-K+1), \qquad
Q_{\text{tiled}} \approx 4\left(L\,\frac{T+K-1}{T} + (L-K+1) + K\left\lceil\tfrac{L-K+1}{T}\right\rceil\right)
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs (one multiply-add per tap per output) |
| $Q_{\text{naive}}$ | bytes if every tap read $x$ and $w$ from global memory |
| $Q_{\text{tiled}}$ | bytes with tiling: each input read about $(T+K-1)/T$ times (halo overlap), each output written once, the kernel read once per block |
| $T$ | outputs per block (1024) |

At the benchmark size, $W \approx 6.1\times10^9$ FLOP, while
$Q_{\text{tiled}} \approx 4(1.5\text{M}\cdot 3 + 1.5\text{M} + 3\text{M}) \approx 36$ MB.
The intensity is $\approx 170$ FLOP/byte, so the kernel is **compute-bound**
on shared-memory loads and FMAs. It runs at about 1 FMA per shared load. A
larger register tile, e.g. 8 outputs per thread, pushes toward the FMA
roofline.

## Pitfalls

- **Dynamic shared memory size.** Forgetting the third launch argument gives
  0 bytes, and every access is out of bounds. Above 48 KB,
  `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize)` would be needed;
  the maximum here is 20 KB.
- **Convolution vs. correlation.** The reference is `unfold` + `einsum`,
  which does not flip the kernel.
- **Output length.** $L - K + 1$, not $L$. There is no padding.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-4`,
including $K = 1$, $K = L$ (single output) and the maximum $K = 2047$.

## Related

- [2D Convolution](../010-2d-convolution/), [3D Convolution](../011-3d-convolution/),
  [Causal Depthwise Conv1D](../090-causal-depthwise-conv1d/).
- Tensara [1D Convolution](../../tensara/conv-1d/).

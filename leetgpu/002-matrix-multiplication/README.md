---
title: Matrix Multiplication
platform: LeetGPU
upstream: easy/2_matrix_multiplication
url: https://leetgpu.com/challenges/matrix-multiplication
difficulty: easy
tags: [gemm, shared-memory, tiling, register-blocking]
status: solved
---

# Matrix Multiplication

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/matrix-multiplication)

## Problem

Multiply two row-major float32 matrices: $A$ is $M \times N$, $B$ is
$N \times K$, and the $M \times K$ product goes to `C`
($1 \le M, N, K \le 8192$; benchmark $M = 8192$, $N = 6144$, $K = 4096$).
Note the naming: in this problem **$N$ is the shared (inner) dimension** and
$K$ is the number of output columns, which is the reverse of the usual BLAS
convention. The reference is `torch.matmul` with `atol = rtol = 1e-4`.

GEMM is the most important kernel in deep learning, and this problem is the
classic place to learn **data reuse**: the naive kernel is ~30× slower than a
tiled one on the same hardware.

## Formulation

$$
C_{rc} = \sum_{k=0}^{N-1} A_{rk}\, B_{kc}, \qquad 0 \le r < M,\ \ 0 \le c < K
$$

| Symbol | Meaning |
|---|---|
| $M$ | rows of $A$ and of $C$ |
| $N$ | columns of $A$ = rows of $B$ (the reduction dimension) |
| $K$ | columns of $B$ and of $C$ |
| $r,\ c$ | row and column of the output element |
| $k$ | summation index along the inner dimension |
| $A_{rk}$ | element of $A$ stored at offset $rN + k$ (row-major) |
| $B_{kc}$ | element of $B$ stored at offset $kK + c$ |
| $C_{rc}$ | element of $C$ stored at offset $rK + c$ |

### Tiling

Split the three loops into tiles. A thread block owns a
$T_M \times T_K$ output tile, and walks the inner dimension in slices of
width $T_N$:

$$
C_{\text{tile}} = \sum_{s=0}^{\lceil N/T_N\rceil - 1} A[\,r_0 : r_0{+}T_M,\ sT_N : (s{+}1)T_N\,]\ \cdot\ B[\,sT_N : (s{+}1)T_N,\ c_0 : c_0{+}T_K\,]
$$

| Symbol | Meaning |
|---|---|
| $T_M,\ T_K$ | output tile size per block: 64 × 64 here |
| $T_N$ | inner-dimension slice staged in shared memory per step: 16 here |
| $s$ | slice index |
| $r_0,\ c_0$ | top-left corner of the block's tile: $r_0 = 64\,\texttt{blockIdx.y}$, $c_0 = 64\,\texttt{blockIdx.x}$ |
| $X[a{:}b,\ c{:}d]$ | sub-matrix with rows $a \dots b-1$ and columns $c \dots d-1$ |

## Approach

### Parallel Decomposition

| Level | Owns | Size |
|---|---|---|
| Grid | whole $C$ | $\lceil K/64\rceil \times \lceil M/64\rceil$ blocks |
| Block (256 threads) | one 64 × 64 tile of $C$ | 4096 outputs |
| Thread | 4 × 4 outputs, strided by 16 | 16 accumulators in registers |

Thread $(t_x, t_y) = (\texttt{tid} \bmod 16,\ \lfloor\texttt{tid}/16\rfloor)$
owns rows $t_y + 16i$ and columns $t_x + 16j$ for $i, j \in \{0,1,2,3\}$. The
stride of 16 (instead of a contiguous 4 × 4 block) keeps consecutive threads
on consecutive columns, so the final stores to `C` are coalesced.

### Main Loop, per Inner Slice

1. **Stage.** All 256 threads cooperatively copy the 64 × 16 slice of $A$ and
   the 16 × 64 slice of $B$ into shared memory. Out-of-range elements are
   stored as 0, so edge tiles need no special code in the compute loop.
   - The $A$ slice is stored **transposed** (`a_tile[k][r]`). Both operands
     are then read along a contiguous row of shared memory in the inner loop.
   - Rows are padded to 64 + 4 floats to avoid bank conflicts on the
     transposed store.
2. `__syncthreads()`: the slice is complete.
3. **Compute.** For each of the 16 values of $k$, load 4 values of $A$ and 4
   of $B$ into registers and do the $4 \times 4 = 16$ FMAs of an outer
   product.
4. `__syncthreads()`: nobody overwrites the slice while it is still being
   read.

### Why Register Blocking

Per inner step, a thread issues 8 shared-memory loads for 16 FMAs, a ratio of
2 FMAs per load. The classic "one output per thread" tiled kernel needs 2
loads per FMA, which is 4× more shared-memory traffic. Shared-memory bandwidth
is what limits that simpler kernel.

## Cost Analysis

$$
W = 2MNK, \qquad
Q_{\text{naive}} \approx 2MNK \cdot 4, \qquad
Q_{\text{tiled}} \approx 4\left(MN\,\frac{K}{T_K} + NK\,\frac{M}{T_M} + MK\right), \qquad
I_{\text{tiled}} \approx \frac{W}{Q_{\text{tiled}}} \approx \frac{1}{4}\cdot\frac{2}{1/T_K + 1/T_M}
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs: one multiply and one add per $(r, c, k)$ triple |
| $Q_{\text{naive}}$ | global-memory bytes if every FMA fetched both operands from DRAM |
| $Q_{\text{tiled}}$ | bytes with tiling: $A$ is re-read once per column of tiles ($K/T_K$ times), $B$ once per row of tiles ($M/T_M$ times), $C$ written once |
| $I_{\text{tiled}}$ | arithmetic intensity (FLOP/byte), ignoring the $C$ term |
| 4 | bytes per float32 |

With $T_M = T_K = 64$, $I_{\text{tiled}} \approx 16$ FLOP/byte versus
$0.25$ naive, a 64× cut in DRAM traffic. (L2 caching makes the real naive
kernel better than 0.25, but far from 16.) At the benchmark size,
$W = 2 \cdot 8192 \cdot 6144 \cdot 4096 \approx 4.1 \times 10^{11}$ FLOP. On a
GPU with ~20 TFLOP/s of fp32 FMA throughput, the compute floor is ≈ 20 ms.
This kernel reaches a fraction of that; cuBLAS-level performance needs the
further steps in [tutorial 04](../../tutorials/04-tiled-matmul.md) (vectorised
loads, double buffering, warp tiling).

## Pitfalls

- **Dimension naming.** Using $K$ as the inner dimension (BLAS habit) produces
  wrong results whenever $N \ne K$.
- **Edge tiles.** Loading zeros keeps every thread executing the same barriers.
  Skipping the load for out-of-range threads would leave stale data in shared
  memory.
- **Two barriers per slice.** The first makes the data visible. The second
  stops fast threads from overwriting the slice that slow threads are still
  reading.
- **64-bit offsets.** $rN + k$ can reach $8192 \cdot 8192 = 2^{26}$, which is
  fine, but the code uses `size_t` anyway so the kernel is safe for larger
  shapes.

## Verification

All LeetGPU test cases, including non-multiples of 64 such as $M = 1$ or
$N = 3$, pass on [cuemu](../../tools/cuemu/README.md) at `1e-4`. The kernel
is also compile-checked with `nvcc -arch=sm_80`.

## Related

- [GEMM (fp16, tensor cores)](../022-gemm/), [Batched MatMul](../030-batched-matrix-multiplication/),
  [INT8 MatMul](../032-int8-quantized-matmul/).
- Tensara [Square MatMul](../../tensara/square-matmul/), [GEMM + ReLU](../../tensara/gemm-relu/).
- [Tutorial 04 – Tiled matrix multiplication](../../tutorials/04-tiled-matmul.md) and the AMD
  track [05](../../tutorials/05-amd-cdna3-mfma.md)–[07](../../tutorials/07-hipblaslt-tensilelite.md).

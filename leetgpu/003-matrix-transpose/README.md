---
title: Matrix Transpose
platform: LeetGPU
upstream: easy/3_matrix_transpose
url: https://leetgpu.com/challenges/matrix-transpose
difficulty: easy
tags: [shared-memory, coalescing, bank-conflicts, memory-bound]
status: solved
---

# Matrix Transpose

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/matrix-transpose)

## Problem

Transpose a row-major float32 matrix: `input` is $R \times C$ ("rows × cols")
and `output` must be the $C \times R$ matrix $A^{\mathsf T}$
($1 \le R, C \le 8192$; benchmark $R = 7000$, $C = 6000$). No arithmetic is
involved at all. The whole problem is about moving memory efficiently, which
makes it the textbook example of **coalescing** and **shared-memory bank
conflicts**.

## Formulation

$$
\text{out}_{c r} = \text{in}_{r c}, \qquad 0 \le r < R,\ \ 0 \le c < C
$$

$$
\text{addr}_{\text{in}}(r, c) = rC + c, \qquad \text{addr}_{\text{out}}(c, r) = cR + r
$$

| Symbol | Meaning |
|---|---|
| $R$ | number of rows of the input (`rows`) |
| $C$ | number of columns of the input (`cols`) |
| $r,\ c$ | row and column index in the input |
| $\text{in}_{rc}$ | input element at row $r$, column $c$ |
| $\text{out}_{cr}$ | output element at row $c$, column $r$ |
| $\text{addr}(\cdot)$ | linear (element) offset in row-major storage |

The difficulty is visible in the address formulas. If consecutive threads
take consecutive $c$, their reads are contiguous, but their writes are
$R$ elements (i.e. $4R$ bytes) apart. One side is always strided.

## Approach

### Tiled transpose through shared memory

A 32 × 8 thread block handles a 32 × 32 tile. The block origin is
$(r_0, c_0) = (32\,\texttt{blockIdx.y},\ 32\,\texttt{blockIdx.x})$.

1. **Coalesced read.** Thread $(t_x, t_y)$ reads
   $\text{in}_{r_0 + t_y + 8j,\ c_0 + t_x}$ for $j = 0..3$ into
   `tile[t_y + 8j][t_x]`. A warp (fixed $t_y$, $t_x = 0..31$) reads 128
   contiguous bytes.
2. `__syncthreads()`.
3. **Coalesced write.** Swap the roles of the block coordinates. Thread
   $(t_x, t_y)$ writes $\text{out}_{c_0 + t_y + 8j,\ r_0 + t_x}$ from
   `tile[t_x][t_y + 8j]`. A warp now writes 32 consecutive elements of one
   output row.

The transpose happens inside shared memory (row index ↔ column index), where
strided access is cheap. It never happens in DRAM.

### Bank-conflict-free padding

Shared memory has 32 banks of 4 bytes. Word $w$ lives in bank $w \bmod 32$.
In step 3 a warp reads a *column* of the tile, i.e. words
$t_x \cdot P + \text{const}$ for $t_x = 0..31$, where $P$ is the row pitch:

$$
\text{bank}(t_x) = (t_x \cdot P + q) \bmod 32
$$

| Symbol | Meaning |
|---|---|
| $P$ | row pitch of the shared tile in 4-byte words (32 unpadded, 33 padded) |
| $q$ | constant column offset $t_y + 8j$ within the row |
| $\text{bank}(t_x)$ | bank accessed by lane $t_x$ |

With $P = 32$, every lane hits the same bank: a **32-way conflict**, fully
serialised. With $P = 33$ (`tile[32][33]`), the banks are
$(t_x + q) \bmod 32$, all distinct, so the read is conflict-free.

## Cost analysis

$$
Q = 2 \cdot 4RC \ \text{bytes}, \qquad W = 0, \qquad T_{\min} = \frac{Q}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $Q$ | compulsory DRAM traffic: read every element once, write it once |
| $W$ | arithmetic work (none) |
| $\beta$ | DRAM bandwidth |
| $T_{\min}$ | time lower bound |

At the benchmark size, $Q = 2 \cdot 4 \cdot 7000 \cdot 6000 = 336$ MB. A good
transpose runs at close to device-to-device copy bandwidth, so compare your
timing with [Matrix Copy](../031-matrix-copy/). A naive transpose wastes most
of each 32-byte sector on the strided side and typically runs at 3–5× lower
bandwidth.

## Pitfalls

- **Two different bounds checks.** The read phase checks $r < R,\ c < C$ in
  input coordinates. The write phase checks against the *output* shape
  ($C \times R$). Mixing them up corrupts the edges of non-square matrices.
- **Forgetting the padding.** The result is still correct but much slower,
  which is why no test catches it. Only a profiler shows the bank conflicts
  (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared`).
- **Grid orientation.** The grid is $\lceil C/32\rceil \times \lceil R/32\rceil$ (x ↔ columns).

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md), including
$1 \times 1$, $1 \times C$ and $R \times 1$ matrices and sizes that are not
multiples of 32.

## Related

- [Matrix Copy](../031-matrix-copy/): the bandwidth ceiling for this problem.
- [Tutorial 02 – Memory hierarchy & coalescing](../../tutorials/02-memory-hierarchy.md).

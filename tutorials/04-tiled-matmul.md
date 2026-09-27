# Matrix Multiplication 1 – Foundations

> **Part III · Matrix Multiplication** · Prerequisites: [01](01-execution-model.md), [02](02-memory-hierarchy.md) ·
> Next: [Matrix Multiplication 2 – Vectorized Loads](gemm/01-vectorized-loads.md)

Matrix multiplication is the opposite of the kernels in chapters 01–03: it
has plenty of arithmetic per byte, so it *can* be compute-bound, but only if
data is reused from fast memory. This chapter builds the standard
optimisation ladder:

1. naive;
2. shared-memory tiling;
3. register tiling;
4. the techniques that get to 80–90 % of cuBLAS (each with its own page in
   the [Matrix Multiplication roadmap](gemm/README.md)).

Throughout, $C = AB$ with $A$ of size $M\times K$, $B$ of size $K\times N$
and $C$ of size $M\times N$, all row-major FP32.

**You will learn**

- why GEMM can be compute-bound, and how much reuse that requires;
- to analyse a kernel's traffic at every level (DRAM/L2, shared memory,
  registers) with one formula;
- block (shared-memory) tiling, including its barriers and bank behaviour;
- register tiling: the outer-product formulation and why it removes the
  shared-memory bottleneck;
- how the epilogue (scaling, bias, activation) fuses into the kernel;
- the map of the remaining optimisations.

## 1. The Numbers That Matter

### 1.1 Work and Compulsory Traffic

$$
C_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{kj}, \qquad
W = 2MNK, \qquad Q_{\min} = 4\,(MK + KN + MN), \qquad
I_{\max} = \frac{W}{Q_{\min}} \xrightarrow{M = N = K} \frac{n}{6}
$$

| Symbol | Meaning |
|---|---|
| $A, B, C$ | Operands and result |
| $M, N, K$ | Rows of $C$, columns of $C$, reduction length |
| $W$ | Flops (each multiply-add counts as 2) |
| $Q_{\min}$ | Compulsory DRAM bytes if every element were read or written exactly once |
| $I_{\max}$ | Best possible arithmetic intensity; for square $n\times n$ matrices it grows like $n/6$ |

At $n = 4096$, $I_{\max} \approx 680$ flop/byte, far above any GPU's ridge
point (chapter 00). The whole game is to get the *actual* intensity, as
seen from DRAM and from each level of on-chip memory, high enough.

### 1.2 Where the Reuse Comes From

Every element $A_{ik}$ is used $N$ times (once per column of $C$) and every
$B_{kj}$ is used $M$ times. A kernel that fetches an element from DRAM for
every use has intensity 1/4 flop/byte; one that fetches it once and reuses
it from on-chip memory approaches $I_{\max}$. All the techniques in this
chapter are ways of scheduling the computation so that an element, once
loaded into a fast memory, is used as many times as possible before it is
evicted.

A unit of work (a block, a warp or a thread) that owns a
$T_M\times T_N$ tile of $C$ and walks $K$ needs $T_M$ values of $A$ and $T_N$
of $B$ per $k$, and does $T_MT_N$ FMAs with them:

$$
\frac{\text{FMAs}}{\text{values loaded}} = \frac{T_M T_N}{T_M + T_N}
$$

| Symbol | Meaning |
|---|---|
| $T_M, T_N$ | Rows and columns of $C$ owned by the unit |

This one ratio, applied at each level, explains every result below.

## 2. Naive Kernel

### 2.1 One Thread per Output

```cpp
__global__ void matmulNaive(const float* a, const float* b, float* c, int m, int n, int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) acc = fmaf(a[row * k + kk], b[kk * n + col], acc);
    c[row * n + col] = acc;
}
```

### 2.2 Its Intensity

Every FMA loads two floats (8 bytes) and does 2 flops ($T_M = T_N = 1$):

$$
I_{\text{naive}} = \frac{2}{8} = 0.25\ \frac{\text{flop}}{\text{byte}}
$$

| Symbol | Meaning |
|---|---|
| $I_{\text{naive}}$ | Intensity seen by the load/store units |

### 2.3 What the Caches Do for It

Within a warp (32 consecutive `col`, same `row`), the load of
`a[row * k + kk]` is the same address for all lanes (one broadcast
transaction), and the load of `b[kk * n + col]` is 32 consecutive floats
(coalesced). Neighbouring warps and blocks re-read the same rows of $A$ and
columns of $B$, and L1/L2 serve many of those re-reads. So the kernel is not
as slow as 0.25 flop/byte from DRAM would suggest, but it is bound by the
load instruction rate and L1/L2 bandwidth, and typically reaches only 1–5 %
of peak.

## 3. Shared-Memory Tiling

### 3.1 The Idea

Split $C$ into $T\times T$ tiles, one per block of $T\times T$ threads, and
split the $k$ loop into phases of $T$:

$$
C_{\mathcal{I}\mathcal{J}} = \sum_{s=0}^{\lceil K/T \rceil - 1} A_{\mathcal{I},\,\mathcal{K}_s}\; B_{\mathcal{K}_s,\,\mathcal{J}}
$$

| Symbol | Meaning |
|---|---|
| $\mathcal{I}, \mathcal{J}$ | The $T$ rows and $T$ columns of one output tile |
| $\mathcal{K}_s$ | The $s$-th slice of $T$ reduction indices |
| $A_{\mathcal{I},\mathcal{K}_s}$, $B_{\mathcal{K}_s,\mathcal{J}}$ | $T\times T$ sub-matrices staged in shared memory |

![Block tiling: a block owns one tile of C and walks the matching row panel of A and column panel of B one slice at a time](figures/ch04-block-tiling.svg)

### 3.2 The Kernel

```cpp
constexpr int kTile = 32;

// launch: block(kTile, kTile), grid(ceil(n / kTile), ceil(m / kTile))
__global__ void matmulTiled(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float a_tile[kTile][kTile];
    __shared__ float b_tile[kTile][kTile];
    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    float acc = 0.0f;
    for (int k0 = 0; k0 < k; k0 += kTile) {
        // Each thread loads one element of each tile; out-of-range -> 0.
        const int a_col = k0 + threadIdx.x, b_row = k0 + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();
        for (int kk = 0; kk < kTile; ++kk) acc = fmaf(a_tile[threadIdx.y][kk], b_tile[kk][threadIdx.x], acc);
        __syncthreads();   // before the next phase overwrites the tiles
    }
    if (row < m && col < n) c[row * n + col] = acc;
}
```

### 3.3 The Two Barriers

Each phase needs two barriers, for two different hazards:

1. **After the loads** (read-after-write): a thread's FMA loop reads tile
   elements loaded by other threads.
2. **After the FMAs** (write-after-read): the next phase overwrites the
   tiles, which slower threads may still be reading.

Removing the second barrier gives results that are right most of the time,
which is the worst kind of bug. [Matrix Multiplication 3 – Double Buffering](gemm/02-double-buffering.md) shows
how two buffers reduce this to one barrier per phase.

### 3.4 Access Patterns

- Loads of both tiles are coalesced: `threadIdx.x` walks a row of $A$'s
  tile and a row of $B$'s tile.
- In the inner loop, `a_tile[ty][kk]` is the same address for the whole
  warp (a broadcast), and `b_tile[kk][tx]` reads a row (conflict-free).
- Zero-filling out-of-range elements keeps the inner loop free of bounds
  checks: a zero contributes nothing to the sum.

### 3.5 Traffic and the New Bottleneck

Each block loads $2T^2$ floats per phase and performs $T^3$ FMAs with them:

$$
Q_{\text{tiled}} = \underbrace{\frac{MN}{T^2}}_{\text{blocks}}\cdot\underbrace{\frac{K}{T}}_{\text{phases}}\cdot\underbrace{2T^2\cdot 4}_{\text{bytes per phase}} = \frac{8MNK}{T}, \qquad
I_{\text{tiled}} = \frac{2MNK}{8MNK/T} = \frac{T}{4}
$$

| Symbol | Meaning |
|---|---|
| $T$ | Tile width (32 here) |
| $Q_{\text{tiled}}$ | Bytes loaded from global memory (L2/DRAM) in total |
| $I_{\text{tiled}}$ | Global-memory intensity: global traffic drops by a factor of $T$ |

With $T = 32$, $I = 8$ flop/byte from L2/DRAM. The new bottleneck is shared
memory: the inner loop still does **one shared load per FMA** (the
broadcast of `a_tile` is nearly free, the `b_tile` read is not), and an SM
can issue far fewer shared loads than FMAs per cycle (an A100 SM does 64
FP32 FMAs per cycle but reads 32 words of shared memory per cycle). This
kernel typically reaches 10–20 % of peak.

## 4. Register Tiling

### 4.1 The Outer Product

Give each thread a $t_M\times t_N$ patch of outputs held in registers. For
each $k$, a thread loads $t_M$ values of $A$ and $t_N$ values of $B$ from
shared memory and does $t_Mt_N$ FMAs (an outer product):

![Register tiling: per k step a thread loads 4 values of A and 4 of B and does 16 FMAs](figures/ch04-register-tile.svg)

### 4.2 The Numbers

$$
\frac{\text{FMAs}}{\text{shared loads}} = \frac{t_M t_N}{t_M + t_N}, \qquad
I_{\text{L2}} = \frac{2\,B_MB_NB_K}{4\,B_K\,(B_M + B_N)} = \frac{B_MB_N}{2\,(B_M + B_N)}
$$

| Symbol | Meaning |
|---|---|
| $t_M, t_N$ | Outputs per thread along $M$ and $N$ (register tile) |
| $B_M, B_N$ | Outputs per block (block tile) |
| $B_K$ | Depth of one K-slice staged in shared memory |
| $I_{\text{L2}}$ | Flops per byte loaded into shared memory from L2/DRAM |

| Configuration | FMAs per shared load | $I_{\text{L2}}$ (flop/B) |
|---|---|---|
| $T = 32$, 1 output per thread | 1 (0.5 counting both loads) | 8 |
| $64\times64$ block, $4\times4$ per thread | 2 | 16 |
| $128\times128$ block, $8\times8$ per thread | 4 | 32 |

Both ratios are the reuse formula of section 1.2, at the thread level and
at the block level.

### 4.3 The Inner Loop

The Tensara matmul pages use the $64\times64$ / $4\times4$ version
([Tensara – Matrix Multiplication](../tensara/matrix-multiplication/)). Its
inner loop is:

```cpp
#pragma unroll
for (int kk = 0; kk < kTileK; ++kk) {
    float a_frag[4], b_frag[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) a_frag[i] = a_tile[kk][ty + 16 * i];   // A stored transposed
#pragma unroll
    for (int j = 0; j < 4; ++j) b_frag[j] = b_tile[kk][tx + 16 * j];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
}
```

### 4.4 Details That Matter

- **Transposed $A$ tile** (`a_tile[k][m]`): both operands are then read
  along rows of shared memory.
- **Stride-16 ownership** (`ty + 16 * i`, `tx + 16 * j`) instead of four
  adjacent elements: at every store instruction, 16 consecutive lanes
  write 16 consecutive columns (coalesced), and shared reads stay
  conflict-free.
- **Padding** of the shared tiles (`[kTileK][64 + 4]`) avoids conflicts in
  the transposed store.
- **Full unrolling** keeps `acc[4][4]` in registers; a dynamic index would
  force it into local memory.

### 4.5 The Price: Registers

A $t_M\times t_N$ tile needs $t_Mt_N$ accumulators plus $t_M + t_N$ fragment
registers plus addresses: about 40 registers for $4\times4$, about 120 for
$8\times8$. More registers per thread means fewer resident warps
(chapter 01, section 6), which is fine as long as each warp has enough
independent FMAs to hide latency by itself. Beyond $8\times8$ in FP32, the
accumulators no longer fit and the compiler spills.

## 5. The Rest of the Ladder

Each technique below has its own page in the
[Matrix Multiplication roadmap](gemm/README.md), with a complete program that is tested
on the CPU emulator:

| Technique | Why it helps | Page |
|-----------|-----|---|
| `float4` shared loads (`LDS.128`) | 4× fewer shared-load instructions for the fragments | [2 – Vectorized Loads](gemm/01-vectorized-loads.md) |
| Double buffering (two sets of tiles) | Load slice $s+1$ while computing slice $s$; one barrier per slice instead of two | [3 – Double Buffering](gemm/02-double-buffering.md) |
| `cp.async` (sm_80+) / TMA (sm_90) | Global → shared copies without going through registers, asynchronous | [4 – Async Copies](gemm/03-async-copies.md) |
| Warp tiling | A warp owns a $64\times32$ sub-tile; matches the hardware hierarchy block → warp → thread and improves register reuse | [5 – Warp Tiling](gemm/04-warp-tiling.md) |
| Swizzled tile order ("grouped" launch) | Blocks that run together share rows of $A$ and columns of $B$ in L2 | [6 – Tile Swizzling](gemm/05-tile-swizzling.md) |
| Split-K / Stream-K | More parallelism when $M\cdot N$ has too few tiles to fill the GPU | [7 – Split-K and Stream-K](gemm/06-split-k-stream-k.md) |
| Tensor cores (WMMA, `mma.sync`, `wgmma`, CUTLASS/CuTe) | 8–16× the FLOPs for FP16/BF16/TF32/FP8; changes the whole data flow | [8 – Tensor Cores](gemm/07-tensor-cores.md) |
| Production design | Select, fuse, tune, validate and operate GEMM kernels across real workloads | [9 – Production GEMM](gemm/08-production-gemm.md) |

A typical progression on one GPU, as a fraction of cuBLAS FP32:

| Kernel | Fraction of cuBLAS |
|---|---|
| Naive | 1–5 % |
| Shared-memory tiling | 10–20 % |
| $4\times4$ register tiling | 40–60 % |
| $8\times8$, `float4`, double buffering, warp tiling | 80–95 % |

![Typical fraction of cuBLAS FP32 throughput reached by each rung of the ladder](figures/ch04-ladder.svg)

Chapters 05–07 continue the story on AMD hardware with matrix-core
instructions, a hand-written assembly kernel and a kernel generator.

## 6. Fusing the Epilogue

Real GEMMs rarely stop at $AB$. The general form is

$$
C \leftarrow f\bigl(\alpha\,AB + \beta\,C + \mathbf{1}\,b^{\mathsf T}\bigr)
$$

| Symbol | Meaning |
|---|---|
| $\alpha, \beta$ | Scalars (BLAS convention) |
| $b$ | A bias vector of length $N$, added to every row |
| $f$ | An elementwise activation (ReLU, GELU, SiLU, …) |

Everything after the $K$ loop happens while the accumulators are still in
registers, so it costs almost nothing:

```cpp
// After the K loop: acc[i][j] holds (AB) for row r_i, column c_j of this thread.
for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
        const float v = alpha * acc[i][j] + beta * c[r_i * n + c_j] + bias[c_j];
        c[r_i * n + c_j] = fmaxf(v, 0.0f);          // ReLU
    }
```

Running the same operations as a separate kernel would read and write $C$
again ($8MN$ bytes). For the skinny GEMMs of LLM inference that extra
traffic can cost as much as the GEMM itself, which is why libraries expose
"fused epilogues" ([Tensara – GEMM + ReLU](../tensara/gemm-relu/),
TensileLite's activation fusions in chapter 07).

## 7. Checklist

- `threadIdx.x` ↔ column, for coalesced $B$ reads and $C$ writes.
- Zero-fill out-of-range elements when staging edge tiles, instead of
  skipping the load (the FMA loop then needs no bounds checks).
- Two `__syncthreads()` per slice with single buffering: after the loads,
  and before the next loads overwrite the tiles.
- Index with `size_t` once $MK$, $KN$ or $MN$ can exceed $2^{31}$ elements
  (for example a $64\cdot4096 \times 4096$ activation in
  [Tensara – Matmul 3D](../tensara/matmul-3d/)).
- Accumulate in FP32 even when inputs are FP16/BF16.
- Check `-Xptxas -v` for spills after every change to the tile sizes.

## Key Takeaways

1. GEMM does $O(n^3)$ work on $O(n^2)$ data; it is compute-bound only if
   each loaded element is reused many times.
2. A unit owning a $T_M\times T_N$ tile does $T_MT_N/(T_M+T_N)$ FMAs per
   value loaded. Apply this at the block level (shared memory) and the
   thread level (registers).
3. Shared-memory tiling fixes DRAM traffic but leaves one shared load per
   FMA; register tiling fixes that.
4. Barriers protect both directions: data ready (after the loads) and
   buffer free (after the math).
5. Fuse the epilogue while the results are in registers.

## Exercises

1. For a $128\times64$ block tile and $8\times4$ thread tile, compute the
   FMAs per shared load and $I_{\text{L2}}$.

    <details markdown="1"><summary>Answer</summary>

    Thread: $32/12 \approx 2.7$ FMAs per load.
    Block: $I_{\text{L2}} = 128\cdot64 / (2\cdot192) \approx 21.3$ flop/B.

    </details>

2. Remove the second `__syncthreads()` from `matmulTiled` and run it under
   cuemu with `CUEMU_REVERSE=1`. What happens, and why is a GPU run not a
   reliable test?

    <details markdown="1"><summary>Answer</summary>

    Fast threads overwrite the tiles while slow ones still read them, so
    some partial sums use the next slice's data. On a GPU the warps usually
    stay close enough in time that the error appears only occasionally.

    </details>

3. How many registers do the accumulators of a $16\times8$ FP32 thread
   tile need? Why is that a problem?

    <details markdown="1"><summary>Answer</summary>

    128 accumulators plus 24 fragment values plus addresses: over 160
    registers, which limits the SM to 1–2 blocks of 256 threads and usually
    spills. [Matrix Multiplication 8 – Tensor Cores](gemm/07-tensor-cores.md) is the way to
    get more work per register.

    </details>

## Practice

- [LeetGPU – Matrix Multiplication](../leetgpu/002-matrix-multiplication/)
- [LeetGPU – GEMM](../leetgpu/022-gemm/)
- [Tensara – Matrix Multiplication](../tensara/matrix-multiplication/)
- [Tensara – GEMM + ReLU](../tensara/gemm-relu/) (fused epilogue)

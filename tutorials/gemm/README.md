# Matrix Multiplication: A 1–9 Roadmap

> **Part III · Matrix Multiplication** · Start: [Matrix Multiplication 1 – Foundations](../04-tiled-matmul.md)

This standalone path starts with the arithmetic and memory model, builds a
fast tiled kernel one technique at a time, and ends with the decisions needed
to ship GEMM in a real system. Read the chapters in order: each implementation
chapter starts from an earlier program and makes one main change.

| Step | Chapter | Main idea | Program |
|---|---|---|---|
| 1 | [Foundations](../04-tiled-matmul.md) | Arithmetic intensity, shared-memory tiling, register tiling and fused epilogues | Inline kernels |
| 2 | [Vectorized Loads](01-vectorized-loads.md) | 128-bit global and shared accesses with a conflict-free fragment layout | [`01-vectorized.cu`](01-vectorized.cu) |
| 3 | [Double Buffering](02-double-buffering.md) | Overlap the next slice's loads with current math | [`02-double-buffering.cu`](02-double-buffering.cu) |
| 4 | [Async Copies](03-async-copies.md) | `cp.async` multi-stage pipelines and Hopper TMA | [`03-cp-async.cu`](03-cp-async.cu) |
| 5 | [Warp Tiling](04-warp-tiling.md) | Match the block → warp → lane hardware hierarchy | [`04-warp-tiling.cu`](04-warp-tiling.cu) |
| 6 | [Tile Swizzling](05-tile-swizzling.md) | Group tile launches to improve L2 reuse | [`05-tile-swizzle.cu`](05-tile-swizzle.cu) |
| 7 | [Split-K and Stream-K](06-split-k-stream-k.md) | Expose parallelism when there are too few output tiles | [`06-split-k.cu`](06-split-k.cu), [`07-stream-k.cu`](07-stream-k.cu) |
| 8 | [Tensor Cores](07-tensor-cores.md) | WMMA, `ldmatrix`, `mma.sync`, shared-memory swizzles and `wgmma` | [`08-wmma.cu`](08-wmma.cu), [`09-mma-sync.cu`](09-mma-sync.cu) |
| 9 | [Production GEMM](08-production-gemm.md) | Persistent and grouped kernels, fusion, accuracy, tuning, dispatch and measurement | Design guide |

Steps 2–8 include complete tested programs. Diff consecutive programs to see
the code cost of each optimization. Step 9 ties the techniques together and
explains when a library is the better production choice.

## The Hierarchy Every Page Refines

![The GEMM tiling hierarchy: each level stages data for the level below](../figures/gemm-hierarchy.svg)

A fast GEMM is the same idea applied at every level of the memory
hierarchy: stage a tile of $A$ and a tile of $B$ in faster memory and reuse
each element as many times as the tile allows. With a
$T_M\times T_N$ output tile per unit (block, warp or lane) and a reduction
step of $T_K$:

$$
\frac{\text{FMAs}}{\text{elements loaded}} = \frac{T_M T_N T_K}{(T_M + T_N)\,T_K} = \frac{T_M T_N}{T_M + T_N}
$$

| Symbol | Meaning |
|---|---|
| $T_M, T_N$ | Output rows and columns owned by one unit at that level |
| $T_K$ | Reduction depth staged at once (cancels out) |

| Level | Staged in | Tile | Reuse $T_MT_N/(T_M+T_N)$ |
|---|---|---|---|
| Block | Shared memory | $128\times128$ | 64 FMAs per element loaded from L2 |
| Warp | (Warp's view of shared memory) | $64\times32$ | 21.3 FMAs per element read from shared memory |
| Lane | Registers | $8\times8$ | 4 FMAs per element read into registers |

[Matrix Multiplication 1 – Foundations](../04-tiled-matmul.md) builds the
block and lane levels. Step 2 makes loads wide; steps 3 and 4 hide their
latency; step 5 adds the warp level; step 6 makes blocks cooperate through
L2; step 7 keeps all SMs busy when there are few tiles; and step 8 replaces
lane-level FMAs with tensor-core instructions. Step 9 turns that kernel
knowledge into a production dispatch and validation strategy.

## Running the Programs

Every program has the same command line, provided by
[`harness.cuh`](harness.cuh):

```bash
cd tutorials/gemm
nvcc -O3 -arch=sm_80 -std=c++17 04-warp-tiling.cu -o warp_tiling
./warp_tiling                # M = N = K = 4096: time, TFLOP/s, spot check of 256 entries
./warp_tiling 2048 512 8192  # any shape
./warp_tiling --test         # awkward shapes, every entry checked against a CPU reference
```

No GPU? The [cuemu](../../tools/cuemu/README.md) emulator runs every
program's `--test` mode on the CPU, including the `cp.async` pipelines,
`ldmatrix` and `mma.sync` (it implements their documented semantics, so a
wrong fragment index or a missing wait fails there too):

```bash
python3 tools/cuemu/cuemu.py run tutorials/gemm/09-mma-sync.cu -- --test
make gemm-test               # all of them, as CI does
```

The emulator checks correctness only. Timings need a GPU; the numbers quoted
on these pages are typical ranges from the literature, not measurements from
this repository.

## Further Reading

- Simon Boehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance* (2022).
- NVIDIA, [CUTLASS](https://github.com/NVIDIA/cutlass) and its
  `media/docs` (efficient GEMM, CuTe layouts, swizzles).
- NVIDIA, *PTX ISA*, sections on `cp.async`, `ldmatrix`, `mma` and `wgmma`.
- Osama et al., *Stream-K: Work-centric Parallel Decomposition for Dense
  Matrix-Matrix Multiplication on the GPU* (PPoPP 2023).
- Triton, *Matrix Multiplication* tutorial (grouped ordering).

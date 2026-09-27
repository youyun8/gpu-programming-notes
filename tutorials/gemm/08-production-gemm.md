# Matrix Multiplication 9 – Production GEMM

> **Part III · Matrix Multiplication** ·
> Prerequisite: [Matrix Multiplication 8 – Tensor Cores](07-tensor-cores.md) ·
> Roadmap: [Matrix Multiplication 1–9](README.md)

A fast kernel for one square matrix is a useful milestone, but production
GEMM is a dispatch problem. Shapes, data types, layouts, fusions and GPU
architectures vary. The best implementation may be a persistent kernel, a
grouped launch, a library call or a small custom kernel selected for one
shape bucket.

**You will learn**

- when persistent, batched and grouped kernels help;
- how grouped expert GEMM supports mixture-of-experts models;
- how CUTLASS and CuTe represent tiled kernels;
- how to fuse epilogues without losing accuracy;
- how to autotune a bounded set of shape buckets;
- when to use a library and when to maintain a custom kernel;
- how to benchmark, profile and validate a production implementation.

## 1. Start With the Workload

Record the operation before choosing a kernel:

$$
D = f\left(\alpha\,\operatorname{op}(A)\operatorname{op}(B)
          + \beta C + \operatorname{bias}\right)
$$

For every call site, collect:

- $M$, $N$, $K$, batch size and their distributions;
- input, accumulator and output types;
- row-major or column-major layouts, transposes and leading dimensions;
- bias, scaling, activation, residual and quantization operations;
- latency, throughput, workspace and determinism requirements;
- GPU model and whether the process shares the GPU.

Optimize frequent shapes and important tail-latency shapes. A kernel that
wins on $4096^3$ may be poor for decode-time GEMMs with small $M$, or for a
batch containing many unrelated shapes.

## 2. Persistent Kernels

A normal GEMM launches one block for each output tile. A **persistent
kernel** launches about as many blocks as can run concurrently and lets each
block claim several tiles from a work queue:

```cpp
for (int tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
    const TileCoord coord = schedule(tile);
    gemmTile(coord);
}
```

This design can:

- remove wave quantization by balancing several tiles per resident block;
- keep weights or scheduling state hot across work items;
- reduce launch overhead when many small GEMMs are combined;
- support Stream-K-style work distribution from
  [Matrix Multiplication 7](06-split-k-stream-k.md).

Persistence is not automatically faster. Long-lived blocks can monopolize
the GPU, reduce fairness, and make a bad tile schedule more expensive. The
kernel must launch no more blocks than can be resident if blocks wait on one
another. Use occupancy APIs for that limit; do not assume one block per SM.

## 3. Batched, Grouped and Expert GEMM

### 3.1 Batched GEMM

Strided batched GEMM applies the same shape and layout to many matrix pairs:

$$
C_b = A_b B_b,\qquad b = 0,\ldots,B-1
$$

Regular strides make addressing cheap and let a library choose one kernel
for the whole batch. Pointer-array batched GEMM supports unrelated
allocations but adds pointer loads. For very small matrices, assign one warp
or one block to a matrix so that launch overhead does not dominate.

### 3.2 Grouped GEMM

Grouped GEMM accepts a list of problems with different $M$, $N$, $K$,
strides or pointers. A persistent scheduler maps tiles to problems:

```text
problem 0: tiles [0, count0)
problem 1: tiles [count0, count0 + count1)
...
```

Prefix sums locate each problem. For a long list, build a direct tile map or
use a two-level lookup so every tile does not scan the list. Group problems
with compatible data types, layouts and epilogues; otherwise branches in the
main loop and epilogue waste issue slots.

### 3.3 Grouped Expert GEMM

Mixture-of-experts inference routes a different number of tokens to each
expert. If expert $e$ receives $M_e$ tokens, its projection is

$$
Y_e = X_e W_e,\qquad
X_e \in \mathbb{R}^{M_e\times K},\quad
W_e \in \mathbb{R}^{K\times N}.
$$

The $M_e$ values are irregular and may include zero. A grouped expert GEMM
packs non-empty experts into one launch and schedules tiles across all of
them. Good implementations:

- consume routing metadata without copying matrices;
- bucket experts with similar $M_e$ when that improves tile utilization;
- skip empty experts;
- preserve the token-to-output mapping for the following scatter;
- fuse per-expert bias, scales or quantization when possible;
- balance experts dynamically so one large expert does not create a long
  tail.

Measure the full route → GEMM → scatter path. Faster matrix math can lose
overall if metadata conversion or extra packing kernels are added.

## 4. CUTLASS and CuTe

[CUTLASS](https://github.com/NVIDIA/cutlass) supplies tuned GEMM building
blocks and device-level kernels. CuTe, used by current CUTLASS versions,
describes tensors with layouts: a shape plus a mapping from logical
coordinates to storage.

The concepts from this path map directly:

| This path | CUTLASS/CuTe concept |
|---|---|
| Block, warp and instruction tiles | Tiled MMA and collective mainloop |
| Global-to-shared pipeline | Collective copy, `cp.async` or TMA |
| Shared-memory XOR layout | Swizzled CuTe layout |
| Fused output loop | Epilogue collective |
| Tile launch order | Scheduler or thread-block swizzle |

Use CUTLASS templates when library-level control is needed but hand-written
PTX is not. CuTe layouts make address mappings explicit and composable, but
their types can be hard to read. Keep a small reference implementation and
test edge shapes whenever a layout or schedule changes.

## 5. Epilogue Fusion

The accumulators are already in registers at the end of the reduction.
Apply scaling, bias, residuals, activation, clamping or output conversion
before storing:

```cpp
float x = alpha * acc + beta * old_c;
x += bias[col];
x = gelu(x);
out[row * ld + col] = convert<Output>(x);
```

Fusion removes intermediate launches and avoids another read and write of
the output. It matters most when the GEMM is small or skinny.

Do not make one universal epilogue full of runtime branches. Compile a small
set of common combinations and dispatch among them. Keep uncommon chains in
a separate kernel unless profiling shows that another specialization pays
for its code size and maintenance cost.

Tensor-core accumulator layouts may not match the desired store layout.
Choose between direct register stores, a shared-memory exchange, or a
library epilogue based on coalescing and register pressure.

## 6. Mixed Precision and Accuracy

Input type, multiply type, accumulator type and output type are separate
choices. Common policies include FP16 or BF16 inputs with FP32 accumulation,
TF32 multiplication with FP32 accumulation, and scaled FP8 inputs with FP32
or FP16 accumulation.

Check more than one random matrix:

- compare against a higher-precision CPU or GPU reference;
- include zeros, subnormals, large values, NaNs and infinities if the API
  defines their behavior;
- test long $K$, where accumulation error grows;
- test the exact scale and rounding rules used by quantized paths;
- test split reductions, because reassociation changes rounding;
- use both absolute and relative error, with tolerances tied to the data
  type and $K$.

A useful normalized residual is

$$
r = \frac{\lVert C_{\text{test}}-C_{\text{ref}}\rVert_F}
         {\lVert A\rVert_F\lVert B\rVert_F+\epsilon}.
$$

Determinism and accuracy are different requirements. Atomic split-K can be
accurate within tolerance while producing different low bits on each run.
Offer a deterministic workspace reduction when callers require repeatable
results.

## 7. Autotuning and Shape Buckets

No single tile shape wins everywhere. Candidate parameters include:

- block, warp and instruction tile shapes;
- pipeline stage count and shared-memory layout;
- split-K factor or Stream-K schedule;
- persistent grid size and tile order;
- vector width, input type and epilogue specialization.

Keep the search bounded. Reject candidates that exceed register, shared
memory, alignment or architecture limits before compiling or running them.
Verify correctness first, warm up the GPU, then compare several timing
samples.

Production dispatch should not store a table for every exact shape. Define
**shape buckets** from workload behavior, for example:

- small $M$ decode shapes;
- tall or wide rectangular shapes;
- large balanced shapes;
- aligned and unaligned $K$;
- common expert-token ranges.

Tune representative shapes, then validate bucket boundaries. Include a
reliable fallback for shapes outside the tuned set. Cache results by GPU
architecture, software version, data type, layout and epilogue; a result
from one GPU is not a portable truth.

## 8. Libraries or Custom Kernels?

Start with cuBLAS, cuBLASLt, CUTLASS or another maintained vendor library.
Libraries usually provide broad shape coverage, architecture dispatch,
workspace selection, fused epilogues and years of correctness tuning.

A custom kernel is justified when measurements show a material end-to-end
gain from one or more of these:

- an unusual fixed shape or layout;
- a fusion the library cannot express;
- grouped expert scheduling tied to application metadata;
- strict workspace, determinism or latency constraints;
- a new architecture feature not yet exposed by the library.

Keep the library as the fallback and comparison baseline. The custom path
must pay for testing across shapes, devices, drivers and future
architectures. Peak TFLOP/s on one benchmark is not enough.

## 9. Benchmarking and Profiling

Benchmark the operation the application actually runs:

1. Allocate and initialize outside the timed region.
2. Warm up until clocks, caches and lazy library initialization stabilize.
3. Time with GPU events on the same stream; report median and tail latency.
4. Run enough work to measure small kernels accurately, but preserve real
   launch dependencies.
5. flush or preserve caches according to the application, and state which;
6. compare identical math, types, fusions and determinism policies;
7. verify outputs after timing.

Report useful throughput as

$$
\text{TFLOP/s} = \frac{2MNK}{t\cdot10^{12}},
$$

but also report latency and end-to-end time. For grouped GEMM, sum
$2M_iN_iK_i$ over all problems.

Use Nsight Systems to find launch gaps, synchronization and overlap. Use
Nsight Compute to inspect achieved occupancy, tensor-core utilization,
global and shared-memory throughput, L2 hit rate, bank conflicts, stalls and
register spills. Profile a small representative set: detailed profiling
changes timing and can serialize kernels.

Interpret counters with the model from earlier chapters. Low tensor-core
utilization may come from too few tiles, pipeline stalls, a costly epilogue
or load imbalance; it does not by itself identify the fix.

## 10. Production Checklist

- [ ] The API defines shapes, layouts, strides, aliases, data types and
      supported epilogues.
- [ ] Empty matrices, edge tiles, odd leading dimensions and large indices
      are correct.
- [ ] Every vectorized path checks alignment and has a safe fallback.
- [ ] FP32 accumulation or another documented accuracy policy is used.
- [ ] Tests cover adversarial values, long $K$, split reductions and every
      fused epilogue.
- [ ] Deterministic and non-deterministic modes are explicit.
- [ ] The dispatcher has a library or general-kernel fallback.
- [ ] Tuning data is keyed by GPU architecture and software version.
- [ ] Workspace size, initialization and stream ownership are documented.
- [ ] Persistent and cross-block protocols cannot deadlock.
- [ ] Benchmarks include production shapes, cold and warm cache behavior,
      median latency, tail latency and end-to-end impact.
- [ ] Profiling shows no accidental spills, bank conflicts or serialization.
- [ ] Library and custom baselines perform exactly the same operation.
- [ ] Unsupported devices fail clearly instead of silently using the wrong
      instruction path.

## Key Takeaways

1. Production GEMM is a workload-aware dispatcher, not one universal kernel.
2. Persistent and grouped scheduling improve utilization for irregular or
   numerous small problems, but require careful load balancing.
3. CUTLASS and CuTe package the same tiling, pipeline and layout ideas used
   throughout this path.
4. Fusion, mixed-precision policy and shape-bucket tuning must be validated
   together.
5. Prefer a maintained library until an end-to-end benchmark proves that a
   custom path is worth owning.

## Exercises

1. Design shape buckets for a workload containing prefill GEMMs, decode
   GEMMs and 64 experts. State the representative shapes and fallback.
2. Add bias and GELU to a tensor-core epilogue. Compare fused and separate
   kernels for a large square GEMM and a small-$M$ GEMM.
3. Build a grouped scheduler for ten problems with different tile counts.
   Compare a prefix-sum lookup with a direct tile-to-problem map.
4. Tune block shape, stage count and split-K factor for three shapes. Record
   correctness, registers, shared memory, median latency and the winning
   configuration.
5. Compare a custom kernel with cuBLASLt or CUTLASS using identical data
   types and epilogues. Identify whether any gain comes from matrix math,
   fusion or scheduling.

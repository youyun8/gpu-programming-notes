# 07 – hipBLASLt and TensileLite: GEMM Kernels Written by a Program

aiter (chapter 06) hand-writes a few dozen GEMM kernels.
[hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/) ships **thousands**: ROCm's `libhipblaslt` holds one set of code objects per GPU
architecture, and PyTorch uses it for `torch.matmul` on MI300 by default.
Nobody writes those by hand. They come from **TensileLite**, a Python program
that takes a list of parameters and emits a complete assembly kernel for
every combination, and then benchmarks the combinations to decide which to
ship.

This chapter explains:
- what those parameters mean;
- how they map onto the techniques from chapters 05–06;
- how hipBLASLt picks a kernel at run time;
- how you tune it for your own shapes.

> **Where the code lives.** hipBLASLt used to be a standalone repository,
> `ROCm/hipBLASLt`; it is now retired to a `develop_deprecated` branch.
> Development continues in the [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries)
> monorepo under `projects/hipblaslt`. The generator lives in
> `tensilelite/Tensile/`:
> - `KernelWriterAssembly.py`, `KernelWriter.py`: the generator.
> - `Components/`: pluggable pieces such as `SIA.py`, `StreamK.py`, `GSU.py`,
>   `LocalRead.py` and `MAC_*.py`.
> - `Common/ValidParameters.py`: every parameter, with comments.
> - `SolutionStructs/`: validation and naming.
>
> File references below are to those paths.

## 1. From "a GEMM" to "a Solution"

A TensileLite **solution** is one point in a large design space:

| Group | Parameters | Chapter 05/06 equivalent |
|-------|------------|--------------------------|
| Tile shape | `MatrixInstruction`, `DepthU` | MFMA shape, wave tile, waves per workgroup, K step |
| Global → LDS | `PrefetchGlobalRead` (PGR), `DirectToLds` (DTL), `GlobalReadVectorWidth`, `BufferLoad` | Prefetch depth, direct-to-LDS loads |
| LDS | `1LDSBuffer`, `LdsPadA/B`, `LdsBlockSizePerPad`, `TransposeLDS` | Double buffering, bank-conflict padding |
| LDS → registers | `PrefetchLocalRead` (PLR), `ClusterLocalRead` | Register double buffering (`a[0:63]` / `a[64:127]`) |
| Scheduling | `ScheduleIterAlg` (SIA), `GlobalReadPerMfma`, `LocalWritePerMfma` | Interleaving loads between MFMAs |
| Work decomposition | `GlobalSplitU` (GSU), `GlobalSplitUAlgorithm`, `StreamK` | Split-K, Stream-K |
| Cache behaviour | `WorkGroupMapping` (WGM), `WorkGroupMappingXCC` (WGMXCC), `StaggerU*` | Tile order, XCD placement, DRAM channel spreading |
| Epilogue | `StoreRemapVectorWidth`, `StoreVectorWidth`, activation / bias / scaling fusions | Coalesced stores |

### 1.1 `MatrixInstruction`: The Tile Hierarchy in 9 Numbers

The comment in `ValidParameters.py` explains the 9-number format:

```
[32, 32, 1, 2,   1,   4, 1,   2, 2]
 ^^^^^^^^^^^^    ^    ^^^^    ^^^^
 MFMA MxNxKxB  BlkM  WaveTile  Waves
```

- **MFMA** `32x32x1x2` is a 2-block MFMA variant. With `MIBlockM = 1`, each
  instruction covers 32×64.
- **WaveTile** `4×1`: each wave issues 4×1 of those, covering 128×64.
- **Waves** `2×2`: four waves per workgroup, so the **macro tile** is
  (32·4·2) × (64·1·2) = **256×128**.

In general, with the 9 numbers written as
$[m, n, k, b,\ \beta_M,\ w_M, w_N,\ W_M, W_N]$:

$$
\text{MT}_0 = m\,\beta_M\,w_M\,W_M, \qquad
\text{MT}_1 = n\,\frac{b}{\beta_M}\,w_N\,W_N, \qquad
\text{threads} = 64\,W_M W_N
$$

| Symbol | Meaning |
|---|---|
| $m, n, k$ | MFMA shape (e.g. 32, 32, 1) |
| $b$ | number of blocks the MFMA computes at once (multi-block variants), 1 for most |
| $\beta_M$ | `MIBlockM`: how many of those $b$ blocks are stacked along M (the rest go along N) |
| $w_M, w_N$ | WaveTile: MFMA tiles per wave along M and N |
| $W_M, W_N$ | waves per workgroup along M and N |
| $\text{MT}_0, \text{MT}_1$ | macro tile (workgroup tile) along M and N |

For the example: $\text{MT}_0 = 32\cdot1\cdot4\cdot2 = 256$ and
$\text{MT}_1 = 32\cdot2\cdot1\cdot2 = 128$, 256 threads.

For the gfx942 bf16 kernels you mostly see `16x16x16` or `32x32x8` MFMAs,
written as `[16,16,16,1, 1, …]`.

Choosing a larger WaveTile is how TensileLite raises the MFMA-per-byte ratio
that chapter 05's teaching kernel lacked. It costs accumulator registers:
- A 128×64 wave tile in fp32 is 8192 values over 64 lanes, which is 128 AGPRs
  per lane.
- That is why fast kernels run at one or two waves per SIMD.

The accumulator cost and the reuse both follow from the wave tile
$T_M\times T_N$ (the MFMA tile times the WaveTile):

$$
r_{\text{acc}} = \frac{T_M T_N}{64}, \qquad
\frac{\text{MFMAs}}{\text{operand fetches}} = \frac{w_M w_N}{w_M + w_N}
$$

| Symbol | Meaning |
|---|---|
| $T_M, T_N$ | output elements per wave along M and N |
| $r_{\text{acc}}$ | fp32 accumulator registers per lane |
| $w_M, w_N$ | MFMA tiles per wave; each A fragment is reused $w_N$ times and each B fragment $w_M$ times |

A $4\times4$ WaveTile of 16x16 MFMAs ($T_M = T_N = 64$) needs 64
accumulator registers per lane and reuses every operand fragment 4 times;
$128\times64$ needs 128.

**`DepthU`** is the K extent of one main-loop iteration: 64 in the aiter
kernel, and typically 32–128 for 16-bit types.

### 1.2 Global Reads: PGR, DirectToLds

`PrefetchGlobalRead`, as described in `ValidParameters.py`:

- **PGR=0:** no prefetch. Load, wait, write to LDS, compute.
- **PGR=1:** double-buffer the path *global → VGPRs → LDS*. Needs twice the LDS
  plus staging VGPRs.
- **PGR=2:** issue *another* global prefetch while the staged data is being
  written to LDS, so two tiles are in flight.

This is exactly what aiter's `pf3` kernels hard-code, and what `vmcnt(N)`
expresses in chapter 06.

`DirectToLds=1` uses `buffer_load … lds` (chapter 06, section 4):
- It removes the staging VGPRs and the `ds_write`s. The comment notes it
  "can save 33 VGPRs" in one configuration.
- Constraints:
  - each lane moves 4 bytes (`GlobalReadVectorWidth · bpe = 4`);
  - `M0` must hold the LDS address;
  - for some layouts, `TransposeLDS=1`.

### 1.3 LDS and Local Reads

- **`1LDSBuffer`** chooses one LDS buffer instead of two, trading overlap for
  capacity: bigger tiles, or higher occupancy. With SIA3 it works only with
  PGR.
- **`LdsPadA/B`** and **`LdsBlockSizePerPad`** insert padding to break bank
  conflicts. This is the same idea as the `+8` row padding in
  `mfma_gemm.hip`, but searched instead of guessed.
- **`PrefetchLocalRead=n`** keeps *n* iterations' worth of `ds_read` results
  in registers ahead of the MFMAs. This is the `a[0:63]` / `a[64:127]` swap
  from chapter 06, generalised.

### 1.4 Scheduling: `ScheduleIterAlg`

The generator first emits the four instruction streams of one loop iteration
separately:
1. global reads and their pointer increments;
2. local writes;
3. local reads;
4. MFMAs.

`KernelWriter.makeSchedule` then asks the `SIA` component how to merge them:

| SIA | Strategy |
|-----|----------|
| 0 | No interleaving: global reads, local reads, local writes, then all MACs |
| 1 / 2 | Older heuristics that interleave per local-read iteration |
| **3** | **MFMA-centric:** place memory instructions *between* MFMAs at a controlled density |

Under SIA=3, `GlobalReadPerMfma` and `LocalWritePerMfma` (0.01–32) control the
density. `0.1` means one global read every 10 MFMAs.

Clustering global reads back to back improves memory efficiency, but a full
vector-memory FIFO blocks *all* issue, MFMAs included, so the density is
tuned. The resulting code has the same shape as the hand-written aiter loop:
an MFMA, one or two loads, an MFMA, and so on.

The generator also computes every `s_waitcnt` itself. It knows how many
loads it placed between each producer and its consumer, capped by the
hardware's `MaxVmcnt`, so it can emit the tightest safe count.

### 1.5 Work Decomposition: GSU and Stream-K

Suppose `M·N / (MT0·MT1)` output tiles is much smaller than 304 CUs, for
example M = 128 during decode. There are two ways to use the idle CUs.

**GlobalSplitU (GSU).** Split K into GSU slices. The partial results are
combined in one of three ways:

| `GlobalSplitUAlgorithm` | How partials are combined |
|-------------------------|---------------------------|
| `SingleBuffer` | Atomic accumulation into one buffer, like aiter's `global_atomic_add_f32` |
| `MultipleBuffer` | Each slice writes its own buffer; a second kernel reduces them |
| `MultipleBufferSingleKernel` | Separate buffers, but the last workgroup to arrive reduces them in the same kernel, using a synchroniser/semaphore like aiter's |

`GSU=-1` lets the runtime choose.

**Stream-K** ([Osama et al., 2023](https://arxiv.org/abs/2301.03598)):
- Launch about one workgroup per CU.
- Give each workgroup an *equal share of the total MAC-loop iterations*
  across all tiles, so a workgroup may finish one tile and start the middle
  of the next.
- Partial tiles are fixed up either through a workspace (deterministic) or
  with atomics.

The balance Stream-K achieves, in formulas:

$$
L = T\left\lceil \frac{K}{\text{DepthU}} \right\rceil, \qquad
L_g \in \left\{ \left\lfloor \frac{L}{G} \right\rfloor,\ \left\lceil \frac{L}{G} \right\rceil \right\}, \qquad
\eta_{\text{SK}} = \frac{L}{G\,\lceil L/G \rceil}
\quad\text{vs.}\quad
\eta_{\text{tile}} = \frac{T}{G\,\lceil T/G \rceil}
$$

| Symbol | Meaning |
|---|---|
| $T$ | number of output (macro) tiles |
| DepthU | K per main-loop iteration |
| $L$ | total MAC-loop iterations in the whole GEMM |
| $G$ | number of Stream-K workgroups (about the CU count) |
| $L_g$ | iterations assigned to workgroup $g$ |
| $\eta_{\text{SK}}, \eta_{\text{tile}}$ | fill efficiency of Stream-K and of one-workgroup-per-tile |

Because $L \gg G$, $\eta_{\text{SK}}$ is essentially 1, whereas
$\eta_{\text{tile}}$ can be as low as $\sim 50\%$ when $T$ is slightly above
a multiple of $G$. The price is the fix-up of tiles shared by two
workgroups.

This removes the "last wave is 10% full" quantisation problem. Because one
kernel covers many shapes well, it also shrinks the library. hipBLASLt
exposes it through environment variables:

```bash
export TENSILE_SOLUTION_SELECTION_METHOD=2   # 0 = standard tuned library (default), 2 = Stream-K library
export TENSILE_STREAMK_DYNAMIC_GRID=3        # 0 = all CUs; 3 = analytical model picks the grid (default)
export TENSILE_STREAMK_FIXED_GRID=64         # force 64 workgroups (leave CUs for concurrent kernels)
export TENSILE_STREAMK_MAX_CUS=128           # cap CUs used
```

Precedence is `FIXED_GRID > DYNAMIC_GRID > MAX_CUS > GRID_MULTIPLIER`.

### 1.6 Cache-Aware Tile Order: WGM, WGMXCC, StaggerU

- **`WorkGroupMapping` (WGM)** reorders workgroup IDs so that the tiles in
  flight at once form a box of height WGM in C. Tiles in a box share A-row
  and B-column panels in L2. The formula is
  `wgSerial = wg0 + (wg1 % WGM) · nwg0`.
- **`WorkGroupMappingXCC` (WGMXCC)** undoes MI300's round-robin placement of
  workgroup *i* on XCD *i % 8*. It remaps IDs so that *consecutive logical
  tiles run on the same XCD* and share its 4 MiB L2.
  `WorkGroupMappingXCCGroup` sets the group size, and `-1` means "CU count".
- **`StaggerU`** / `StaggerUStride` / `StaggerUMapping` start each
  workgroup's K loop at a different offset, rotating through K.
  - This matters when K is a large power of two: every tile would otherwise
    begin on the same DRAM channel.
  - `StaggerUMapping` selects which workgroup index drives the offset:
    wg0, wg1, wg2 or the serial ID.

The idea behind WGM is the same as the "grouped" launch order used by
Triton and CUTLASS. Written in that common form, the serial launch index
$s$ is mapped to the tile $(w_0', w_1')$ that the workgroup computes:

$$
s = w_0 + w_1\,n_0, \qquad
w_1' = g\left\lfloor \frac{s}{g\,n_0} \right\rfloor + (s \bmod g), \qquad
w_0' = \left\lfloor \frac{s \bmod g\,n_0}{g} \right\rfloor
$$

| Symbol | Meaning |
|---|---|
| $w_0, w_1$ | launch-order workgroup indices |
| $n_0$ | number of tiles along dimension 0 |
| $g$ | box height (WGM) |
| $s$ | serial launch order |
| $w_0', w_1'$ | the tile actually computed |

Consecutive workgroups first walk down $g$ tile rows, then move one tile
column over, so the workgroups in flight cover a $g$-tall box and share
$g$ row panels of A plus a few column panels of B in L2. TensileLite's own
formula differs in detail (and WGMXCC adds the XCD remap on top), but the
reuse argument is the same.

WGM, WGMXCC, StaggerU and GSU are cheap to change at run time. They are
packed into the kernel arguments rather than compiled in:

```
internalArgs  (32 bit): input type | StaggerU (3-bit mapping, 5-bit shift, 8-bit value)
                        | GSU control (GSUC, GSUWGMRR, 14-bit GSU)
internalArgs1 (32 bit): WGMXCCG (10) | WGMXCC (6) | WGM (16, signed)
```

This layout is documented in `Components/README.md` as kernel-argument
"Version 2". One code object can therefore serve many tuned variants.

## 2. Reading a Kernel Name

Kernel names are generated in `SolutionStructs/Naming.py`:

1. Start with the problem type.
2. Add `MT<MT0>x<MT1>x<DepthU>` and `MI<M>x<N>x<B>`.
3. For every required parameter, append its *uppercase letters* followed by
   its value.

So a name fragment such as

```
…_MT256x256x64_MI16x16x1_SN_…_DTL1_…_PGR2_PLR1_…_SIA3_…
```

decodes as:

| Fragment | Meaning |
|----------|---------|
| `MT256x256x64` | macro tile 256×256, DepthU 64 |
| `MI16x16x1` | 16×16 MFMA, 1 block |
| `DTL1` | DirectToLds |
| `PGR2` | PrefetchGlobalRead=2 |
| `PLR1` | PrefetchLocalRead=1 |
| `SIA3` | ScheduleIterAlg=3 |

Other parameters abbreviate the same way:
- `1LDSBuffer` → `LDSB`
- `WorkGroupMapping` → `WGM`
- `StaggerU` → `SU`
- `GlobalSplitU` → `GSU`

When you profile PyTorch on MI300 and see a `Cijk_…` kernel, this is how you
read what it does.

## 3. How hipBLASLt Chooses at Run Time

1. **Library logic files.** YAML files per architecture and data type are
   produced by benchmarking. They list solutions and the problem sizes where
   each one won.
2. **Heuristic.** `hipblasLtMatmulAlgoGetHeuristic` returns a ranked list for
   the problem, from the "standard grid" (exact tuned sizes), the
   "free-size" libraries, or the Stream-K library (see
   `TENSILE_SOLUTION_SELECTION_METHOD`).
3. **Solution index.** Every solution has a stable index *within a library
   build*. You can request one explicitly with `--algo_method index` in
   `hipblaslt-bench`, or through the extension API.

## 4. Tuning hipBLASLt for Your Shapes

This is the offline tuning flow from `docs/how-to/how-to-use-hipblaslt-offline-tuning.rst`:

```bash
# 1. Log the GEMMs your application issues, as ready-to-run bench commands
export HIPBLASLT_LOG_MASK=32
python my_model.py 2> gemms.log        # prints: hipblaslt-bench --api_method c -m … -n … -k … --algo_method index --solution_index …

# 2. Tune: benchmark every applicable solution, record the winner
export HIPBLASLT_TUNING_FILE=tuning.txt
hipblaslt-bench <one logged line>       # repeat for each unique line; iters/cold_iters default to 1000

# 3. Use: override default selection with the tuned winners
unset HIPBLASLT_TUNING_FILE
export HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt
python my_model.py
```

Two warnings apply:
- Solution indices are only valid for **the same library build and the same
  architecture**. Re-tune after upgrading ROCm.
- `HIPBLASLT_TUNING_USER_MAX_WORKSPACE` limits the chosen solutions to the
  workspace your application actually provides, which matters for GSU and
  Stream-K.

Framework-level alternatives:
- **PyTorch TunableOp:** `PYTORCH_TUNABLEOP_ENABLED=1` tries the hipBLASLt and
  rocBLAS candidates per shape at run time and caches the result in a CSV.
- **aiter's `gemm_a16w16_tune.py --with-hipblaslt`** puts hipBLASLt solutions
  in the same race as its asm, triton and other backends (chapter 06). The
  winner's `solidx` goes into `bf16_tuned_gemm.csv` with
  `libtype=hipblaslt`.

## 5. Generating Your Own Kernels with TensileLite

TensileLite can also be run directly on a YAML config that lists fork
parameters and problem sizes. Example configs live under
`tensilelite/HostLibraryTests/configs/` (for example
`mixed_configs/aquavanjaram_*.yaml`; "aquavanjaram" is gfx942). The format
changes between releases, so start from a config in your own checkout.
TensileLite then:

1. Enumerates every valid combination: validity is checked by
   `SolutionStructs/Validators`, and `MatrixInstruction` is checked by the
   `TensileLogic` program.
2. Generates and assembles each kernel.
3. Benchmarks each one on your GPU.
4. Writes library logic that hipBLASLt can load.

This is the "handcraft by search" half of AMD's approach. aiter's `.co`
kernels are the "handcraft by hand" half. Both end up with the same loop
structure you traced in chapter 06.

## 6. Summary: The AMD GEMM Playbook

| Technique | aiter asm (ch. 06) | TensileLite parameter |
|-----------|-------------------|-----------------------|
| Big per-wave tiles, 1 wave per SIMD | 16×128 per wave, 512 registers | `MatrixInstruction` WaveTile, `MaxOccupancy` |
| Direct-to-LDS | `buffer_load_dword … lds` | `DirectToLds` |
| Pre-arranged weights | B pre-shuffled into AGPRs | `HIPBLASLT_ORDER_COL16_4R8` (bf16/fp16) / `COL16_4R16` (fp8) matrix orders for A on gfx94x |
| Multi-stage prefetch | `vmcnt(18)`, LDS ping-pong | `PrefetchGlobalRead`, `1LDSBuffer` |
| Register double buffering | `a[0:63]` ↔ `a[64:127]` | `PrefetchLocalRead` |
| MFMA/memory interleaving | by hand | `ScheduleIterAlg=3`, `GlobalReadPerMfma` |
| Split-K / Stream-K | z-grid + atomics + semaphore | `GlobalSplitU*`, `StreamK` |
| L2/XCD-aware tile order | – | `WorkGroupMapping`, `WorkGroupMappingXCC`, `StaggerU` |
| Per-shape selection | tuned CSV + heuristic | library logic + heuristic + offline tuning |

## Exercises

1. Take the `hipblaslt-bench` line for a 4096×4096×4096 bf16 GEMM:
   - Run it with `--algo_method heuristic --requested_solution 10 --print_kernel_info`.
   - Decode the top three kernel names with section 2.
   - Which parameters differ?
2. Compare `TENSILE_SOLUTION_SELECTION_METHOD=0` and `=2` on
   M ∈ {1, 16, 128, 1000} with N = K = 8192. Explain the differences with the
   wave-quantisation argument from section 1.5.
3. On an MI300X, run the same GEMM with `WorkGroupMappingXCC` set to 1 and to
   8, if you can find solutions that differ only in that parameter. Measure the
   L2 hit rate with `rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum`.

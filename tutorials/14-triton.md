# 14 – Triton: Block-Level GPU Programming in Python

> **Part V · Tools & Publishing** · Prerequisites: [03](03-parallel-reduction.md),
> [04](04-tiled-matmul.md), [13](13-softmax-attention.md) ·
> Programs: [`examples/14-triton/`](examples/14-triton/test_kernels.py) ·
> Next: [08 – Deploying This Site](08-deploying-this-site.md)

CUDA asks you to write the program of *one thread* and to arrange the
cooperation of thousands of them by hand: which thread loads which bytes,
what goes into shared memory, where the barriers go. Triton raises the
level by one step. You write the program of *one block*, in Python, as
operations on whole tiles (load this 64 × 64 tile, multiply these two
tiles, take the maximum along this axis) and the compiler decides how the
tile is spread over threads, how loads are vectorised and coalesced, what
is staged through shared memory, and which instructions (tensor cores,
`cp.async`, TMA) implement it. Most of the techniques of Part III happen
automatically; the ones that remain are exactly the algorithmic choices:
tile sizes, tile order, fusion.

**You will learn**

- the Triton programming model: programs, blocks, masks and pointer
  tiles;
- four complete kernels (vector addition, fused softmax, matrix
  multiplication with grouped ordering and autotuning, FlashAttention)
  and how each maps to the CUDA of earlier chapters;
- which GEMM optimisations of chapter 04 the compiler performs and which
  remain yours;
- how a Triton kernel is compiled, specialised, cached, inspected,
  debugged in the interpreter, benchmarked and profiled;
- when to choose Triton, CUDA or a library.

## 1. Why a Block-Level Language

### 1.1 What Moves From You to the Compiler

| Concern | CUDA | Triton |
|---|---|---|
| Unit of the program | One thread | One program instance (a block of threads) |
| Data | Scalars in registers | Tensors of static, power-of-two shape |
| Thread ↔ element mapping | You | Compiler (a *layout*) |
| Coalescing, vector width | You (chapter 02, 04.1) | Compiler, from the pointer pattern and alignment |
| Shared memory, barriers | You | Compiler |
| Bank-conflict swizzles | You (04.5) | Compiler |
| Multi-stage load pipeline | You (04.2, 04.3) | Compiler, `num_stages` |
| Tensor-core instructions | You (04.7) | Compiler, from `tl.dot` |
| Tile sizes, grid, tile order | You | You |
| Fusion (what one kernel does) | You | You |

The trade: you give up control over the per-thread code (so a few
techniques, such as hand-written warp specialisation or register-level
tricks, are harder or impossible), and get kernels that are
5–10× shorter and portable across NVIDIA and AMD GPUs.

![CUDA describes one thread and you choose the mapping; Triton describes one block and the compiler lays it out over the warps](figures/ch14-model.svg)

### 1.2 Setup

```bash
pip install torch triton          # Triton ships with PyTorch's CUDA wheels too
cd tutorials/examples/14-triton
python3 test_kernels.py           # checks all four kernels against PyTorch
python3 test_kernels.py --bench   # and times them (GPU only)
```

Without a GPU, `test_kernels.py` sets `TRITON_INTERPRET=1` before
importing Triton. The **interpreter** runs each program instance
sequentially with NumPy: slow, but it executes the same indexing, masks and
arithmetic, so it is the Triton counterpart of this repository's CUDA
emulator and what CI uses. On ROCm, the same code runs on AMD Instinct
GPUs with the ROCm build of PyTorch.

## 2. The Programming Model

### 2.1 Programs and the Grid

A Triton kernel is a Python function decorated with `@triton.jit`. It is
launched on a grid of **program instances**, `kernel[grid](arg0, arg1, ...)`, where
`grid` is a tuple of up to three sizes, or a function of the kernel's
compile-time parameters:

```python
grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
kernel[grid](x, y, out, n, BLOCK=1024)
```

Inside the kernel, `tl.program_id(axis)` is the program's index (CUDA's
`blockIdx`) and `tl.num_programs(axis)` the grid size (`gridDim`). There is
no `threadIdx`: a program is one block, and how many threads execute it is
a launch option, `num_warps` (default 4).

### 2.2 Blocks: Static, Power-of-Two Tensors

Values inside a kernel are scalars or **blocks**: tensors whose shape is
known at compile time. `tl.arange(0, BLOCK)` creates the vector
`[0, 1, …, BLOCK−1]`; `BLOCK` must be a `tl.constexpr` and a power of two.
Operations are element-wise with NumPy broadcasting (`x[:, None]`,
`y[None, :]`); reductions take an axis (`tl.sum`, `tl.max`, `tl.argmax`);
`tl.dot` multiplies two 2-D blocks.

Every distinct value of a `constexpr` compiles a separate kernel, which is
why sizes are passed as keywords: `BLOCK=1024`.

### 2.3 Pointers, Loads, Stores and Masks

Tensor arguments arrive as pointers to their first element. Adding a block
of offsets to a pointer gives a **block of pointers**, and
`tl.load`/`tl.store` read and write all of them at once:

```python
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < n                               # the tail guard, for the whole block
x = tl.load(x_ptr + offs, mask=mask, other=0.0)
tl.store(out_ptr + offs, x, mask=mask)
```

Masked-off lanes neither read nor write; `other` is the value a masked
load returns (choose the identity of what follows: 0 for a sum, −∞ for a
maximum).

Offsets are in *elements*: Triton multiplies by the element size itself.
They are 32-bit integers unless an argument makes them 64-bit; for tensors
of more than $2^{31}$ elements, cast with `pid.to(tl.int64)` first.

### 2.4 Two-Dimensional Tiles

A 2-D tile of pointers is the broadcast sum of a column of row offsets and
a row of column offsets:

![rows[:, None] * S plus cols[None, :] broadcasts to a BLOCK_M × BLOCK_N tile of addresses; masks are built the same way](figures/ch14-pointer-block.svg)

$$
\text{ptr}_{rq} = \text{base} + \text{row}_r \cdot s_0 + \text{col}_q \cdot s_1,
\qquad r < B_M,\ q < B_N
$$

| Symbol | Meaning |
|---|---|
| $\text{base}$ | Pointer to element (0, 0) of the matrix |
| $\text{row}_r,\ \text{col}_q$ | The r-th entry of `rows` and the q-th entry of `cols` |
| $s_0,\ s_1$ | Row and column strides of the matrix, in elements (`x.stride(0)`, `x.stride(1)`) |
| $B_M,\ B_N$ | Tile shape (`BLOCK_M`, `BLOCK_N`) |

Passing both strides makes the same kernel work for row-major, transposed
and sliced matrices. The compiler emits vectorised loads only along a
dimension it can prove contiguous and aligned; it learns this from the
access pattern and from specialising on argument values (section 7.1).
When a stride is always 1, leaving it out of the signature, as the
softmax kernel does, makes the contiguity explicit.

### 2.5 What Is Not in the Language

- **No shared memory or barriers** in normal code: data exchange inside a
  program happens through block operations (`tl.sum`, `tl.dot`,
  `tl.trans`, reshapes), which the compiler implements with shuffles or
  shared memory as needed.
- **No communication between programs** except through global memory and
  atomics (`tl.atomic_add`, `tl.atomic_cas`, …), as in CUDA.
- **No dynamic shapes** inside a kernel: a row of run-time length is
  handled with a power-of-two block and a mask, or a loop over blocks.

## 3. Kernel 1: Vector Addition

[`vector_add.py`](examples/14-triton/vector_add.py) is the whole model in
ten lines:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                  # blockIdx.x
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # a vector of BLOCK indices
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

The grid has $\lceil n / B \rceil$ programs. With `BLOCK = 1024` and
`num_warps = 4`, each of the 128 threads owns 8 elements, which the
compiler loads as two 16-byte vectors per thread: the float4 loads of
chapter 04.1, without writing them. The kernel reaches the same bandwidth
as a good CUDA copy; its whole cost is $12n$ bytes (chapter 00, section 7).

## 4. Kernel 2: Fused Softmax

[`softmax.py`](examples/14-triton/softmax.py) computes a row-wise softmax
with one program per row:

```python
@triton.jit
def softmax_kernel(in_ptr, out_ptr, n_cols, in_stride, out_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(out_ptr + row * out_stride + cols, e / tl.sum(e, axis=0), mask=mask)
```

### 4.1 What the Compiler Generates

`tl.max(x, axis=0)` is the block reduction of chapter 03: per-thread
partial maxima, a warp shuffle tree, and a shared-memory exchange between
the warps, all derived from one call. The masked lanes load −∞, so they
neither change the maximum nor (since $e^{-\infty} = 0$) the sum.

### 4.2 Why It Is Fast

The row is read once into registers and written once: $2 \cdot 4 \cdot n$
bytes per row, the minimum. The three-kernel version of chapter 13,
section 2 reads it three times. The fused kernel's advantage is not
Triton-specific (chapter 13's online warp-per-row kernel does the same in
CUDA), but in Triton fusion is the natural way to write it.

### 4.3 Limits

`BLOCK = next_power_of_2(n_cols)` must fit in registers. The wrapper uses
more warps for longer rows so each thread holds at most ~32 values up to
16 384 columns;
beyond a few tens of thousands of columns the kernel spills. The fix is
the online softmax of chapter 13: loop over the row in blocks, keeping
$(m, z)$ as the state (exercise 2).

## 5. Kernel 3: Matrix Multiplication

[`matmul.py`](examples/14-triton/matmul.py) computes $C = AB$ with one
program per $B_M \times B_N$ tile of $C$:

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k0 in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, mask=(rows[:, None] < M) & (ks[None, :] + k0 < K), other=0.0)
    b = tl.load(b_ptrs, mask=(ks[:, None] + k0 < K) & (cols[None, :] < N), other=0.0)
    acc = tl.dot(a, b, acc)                    # acc += a @ b on tensor cores
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

This is the structure of chapter 04's tiled kernel. The difference is in
what the compiler does with it.

### 5.1 Chapter 04's Techniques, Revisited

| Technique (chapter) | In Triton |
|---|---|
| Vectorised loads (04.1) | Automatic, when strides and alignment allow |
| Double buffering (04.2) | Automatic: `num_stages` buffers in shared memory |
| `cp.async` / TMA (04.3) | `cp.async`: automatic on Ampere and later for pipelined loads; TMA: through tensor descriptors (`tl.make_tensor_descriptor`) on Hopper and later |
| Warp tiling (04.4) | Automatic: the `tl.dot` layout distributes the tile over `num_warps` warps |
| Shared-memory swizzle (04.5) | Automatic |
| Tile-order swizzle (04.5) | Yours: `GROUP_M` (section 5.2) |
| Split-K / Stream-K (04.6) | Yours: an extra grid axis over K, then a reduction or atomics |
| Tensor cores (04.7) | Automatic, from `tl.dot` |

### 5.2 Grouped Tile Ordering

Programs run roughly in the order of their ids. With a row-major order,
the programs in flight at any moment cover one or two rows of tiles and
therefore *all* tile columns of $B$, so $B$ is read from DRAM repeatedly.
Grouping $G$ tile rows together makes the programs in flight cover a
square-ish patch of $C$ instead:

$$
g = \left\lfloor \frac{p}{G\,T_N} \right\rfloor, \quad
G' = \min(T_M - gG,\ G), \quad
m = gG + \big(p \bmod G T_N\big) \bmod G', \quad
n = \left\lfloor \frac{p \bmod G T_N}{G'} \right\rfloor
$$

| Symbol | Meaning |
|---|---|
| $p$ | Program id |
| $T_M,\ T_N$ | Number of tile rows and tile columns of $C$ |
| $G$ | `GROUP_M`, tile rows per group |
| $g$ | The group of program $p$ |
| $G'$ | Tile rows in this group (the last group may be shorter) |
| $m,\ n$ | The tile of $C$ that program $p$ computes |

With $T_N = 32$ and 64 programs in flight, row-major order touches 2 row
strips of $A$ and 32 column strips of $B$ (34 strips); with $G = 8$ it
touches 8 of $A$ and 8 of $B$ (16 strips), so about half the L2 footprint
and correspondingly more L2 hits. This is the same idea as the tile-order
swizzle of chapter 04.5.

### 5.3 Autotuning

The best tile shape depends on the GPU, the data type and the matrix
sizes. `triton.autotune` compiles the kernel for a list of configurations,
times each the first time a new `key` is seen, and caches the winner:

```python
CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
    ...
]
matmul_kernel_tuned = triton.autotune(configs=CONFIGS, key=["M", "N", "K"])(matmul_kernel)
```

| Parameter | Trade-off |
|---|---|
| `BLOCK_M`, `BLOCK_N` | Larger tiles: more reuse per byte loaded (chapter 04, section 3), more registers, fewer programs |
| `BLOCK_K` | Larger: fewer loop iterations, more shared memory per stage |
| `num_stages` | More stages hide more latency, and cost `num_stages` × tile bytes of shared memory |
| `num_warps` | More warps per program: smaller per-warp tiles, more latency hiding, less ILP |

The shared memory per program is about
$\text{num\_stages} \cdot (B_M + B_N) \cdot B_K \cdot \text{sizeof}$; a
configuration that exceeds the SM's capacity fails to compile and is
skipped.

### 5.4 Precision

For `float32` inputs, `tl.dot` uses TF32 tensor cores by default on
Ampere and later (a 10-bit mantissa, relative error ~$10^{-3}$); the test
script loosens its tolerance on a GPU for that reason. `tl.dot(a, b, acc,
input_precision="ieee")` forces full FP32 at a large cost in speed. For
`float16`/`bfloat16` inputs the accumulator stays `float32`, and the
epilogue (`acc.to(c_ptr.dtype.element_ty)`) converts once at the end.

### 5.5 Epilogue Fusion

Anything element-wise on `acc` before the store is free in memory traffic:
a bias add, an activation (`tl.where(acc > 0, acc, 0.0)`), a scale, a
conversion to FP8. This is where a hand-written Triton GEMM most often
beats a library call followed by separate element-wise kernels.

## 6. Kernel 4: FlashAttention

[`flash_attention.py`](examples/14-triton/flash_attention.py) implements
the forward pass of chapter 13, section 5 for one head. Each program owns
`BLOCK_Q` query rows and streams $K$ and $V$ past them:

```python
for start in range(0, kv_end, BLOCK_KV):
    k = tl.load(k_ptr + kv_rows[:, None] * stride_k + dims[None, :], ...)
    v = tl.load(v_ptr + kv_rows[:, None] * stride_v + dims[None, :], ...)
    s = tl.dot(q, tl.trans(k))                         # scores, on-chip
    s = tl.where(valid, s, -float("inf"))              # padding and causal mask
    m_new = tl.maximum(m, tl.max(s, axis=1))
    m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
    p = tl.exp(s - m_safe[:, None])
    alpha = tl.exp(m - m_safe)
    l = l * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    m = m_new
```

Line by line, this is the online-softmax recurrence of chapter 13:

| Line | Chapter 13 |
|---|---|
| `s = tl.dot(q, tl.trans(k))` | $S = Q K^\top$ for one tile, scale folded into $Q$ |
| `m_new`, `alpha` | The new running maximum and the rescaling factor $e^{m_{\text{old}} - m_{\text{new}}}$ |
| `m_safe` | The guard for rows that have seen only masked keys ($-\infty - (-\infty)$) |
| `l`, `acc` | The running sum and the unnormalised output, rescaled, then updated |
| `acc / l[:, None]` (after the loop) | The final normalisation |

The CUDA version in [`examples/13-softmax-attention.cu`](examples/13-softmax-attention.cu)
spends most of its ~100 lines distributing the tiles over lanes and
staging $K$ and $V$ through shared memory; here both `tl.dot` calls run on
tensor cores and the staging is pipelined by the compiler. Production
kernels add a backward pass, multiple heads and batches as extra grid
axes, and on Hopper warp specialisation, but the core is this loop.

For the causal case, `kv_end` stops the loop at the last key the block's
queries can see, which halves the work, exactly as in the CUDA kernel.

## 7. Compilation, Debugging and Profiling

### 7.1 From Python to Machine Code

![The compiler lowers the decorated Python function through Triton IR (block operations), TritonGPU IR (layouts, shared memory, pipelining) and LLVM IR to PTX or AMDGCN](figures/ch14-compiler.svg)

The first launch with a new combination of constexpr values, argument
dtypes and **specialisations** (among others, Triton checks whether
pointers and integer arguments are divisible by 16, which lets it prove
alignment for vectorised loads) compiles the kernel; later launches hit an in-memory and on-disk cache. The
launch returns a handle whose `asm` dictionary holds every stage:

```python
handle = add_kernel[grid](x, y, out, n, BLOCK=1024)
print(handle.asm["ttgir"])    # layouts chosen by the compiler
print(handle.asm["ptx"])      # look for ld.global.v4.f32 (vectorised loads)
```

The TritonGPU IR is where performance questions are answered: it shows
the layout of every tensor (`#blocked`, `#mma`, `#shared`), where shared
memory is allocated, and how many pipeline stages were created.

### 7.2 Debugging

| Tool | Use |
|---|---|
| `TRITON_INTERPRET=1` | Run on the CPU with NumPy; `print()` and `pdb` work inside the kernel |
| `tl.device_print("x", x)` | Print from the compiled kernel (every program prints: restrict with a mask or a small grid) |
| `tl.static_print`, `tl.static_assert` | Print or check constexpr values at compile time |
| `tl.device_assert(cond, "msg")` | Run-time assertion (enabled with `TRITON_DEBUG=1`) |

The common errors are static: `arange` bounds that are not a power of two,
shapes that do not broadcast, a non-constexpr value where the compiler
needs a constant, and `tl.dot` operands smaller than 16 in some dimension.

### 7.3 Benchmarking and Profiling

`triton.testing.do_bench(fn)` times a callable with warm-up, repetitions
and an L2 flush between runs, and returns milliseconds;
`triton.testing.perf_report` sweeps sizes and plots the results. The
roofline arithmetic of chapter 00 and the profilers of chapter 09 apply
unchanged: a Triton kernel is an ordinary kernel to Nsight Compute
(`ncu -k regex:matmul_kernel python3 test_kernels.py --bench`), and with
`-lineinfo`-style information enabled by default, source attribution
points at the Python lines.

## 8. Triton, CUDA or a Library?

| Situation | Choose |
|---|---|
| A standard GEMM, convolution or attention with standard shapes | A library (cuBLAS, cuDNN, hipBLASLt, FlashAttention) |
| A fused operation that no library has (GEMM + custom epilogue, a new attention variant, a fused norm) | Triton |
| Research code that must run on NVIDIA and AMD | Triton |
| The last 10–20 % on one architecture: warp specialisation, custom pipelines, exotic data movement | CUDA (or CUTLASS/CuTe, chapter 04.7) |
| Algorithms dominated by irregular per-thread control flow (sorting networks, graph traversal) | CUDA |

## Key Takeaways

1. A Triton kernel describes one block with static, power-of-two tensors;
   the compiler maps it to threads.
2. Pointer blocks plus masks replace thread indexing and bounds checks;
   `other` supplies the identity for masked lanes.
3. Coalescing, vectorisation, shared-memory staging, pipelining, swizzles
   and tensor-core instructions come from the compiler; tile sizes, tile
   order and fusion remain your decisions.
4. `triton.autotune` searches tile shapes, `num_warps` and `num_stages` per
   problem size.
5. The interpreter (`TRITON_INTERPRET=1`) tests kernels on a CPU; the
   `asm` stages and Nsight Compute explain their performance.

## Exercises

1. In `add_kernel`, what happens if `BLOCK` is 1000? And if `mask` is
   omitted from the load?

    <details markdown="1"><summary>Answer</summary>

    `tl.arange(0, 1000)` fails to compile: block sizes must be powers of two.
    Without the mask, the last program reads past the end of `x` and `y`
    (undefined values; `compute-sanitizer` reports the out-of-bounds
    reads); the store would still be guarded by its own mask,
    so results may look right while the kernel is wrong.

    </details>

2. Write a softmax kernel for rows too long for registers: loop over the
   row in blocks of `BLOCK` columns, keeping a running maximum and sum
   (chapter 13, section 3), then loop again to write the output.

    <details markdown="1"><summary>Hint</summary>

    Keep per-lane vectors `m` and `z` of shape `(BLOCK,)` across the first
    loop (`m_new = tl.maximum(m, x)`, `z = z * tl.exp(m - m_new) + tl.exp(x - m_new)`,
    with the −∞ guard), combine them at the end with
    `M = tl.max(m, 0)`, `Z = tl.sum(z * tl.exp(m - M), 0)`, and in the
    second loop store `tl.exp(x - M) / Z`. Two reads and one write, for any
    row length.

    </details>

3. For $M = N = K = 4096$ in FP16 with `BLOCK_M = BLOCK_N = 128`,
   `BLOCK_K = 32` and `num_stages = 3`, how much shared memory does a
   program need? How many programs fit on an A100 SM (164 KB)?

    <details markdown="1"><summary>Answer</summary>

    $3 \cdot (128 + 128) \cdot 32 \cdot 2 = 49\,152$ bytes = 48 KB, so three
    programs fit by shared memory. With `num_warps = 8` (256 threads) and a
    128 × 128 FP32 accumulator (64 values per thread) plus operands, the
    register file usually limits it to one or two.

    </details>

4. Add a fused ReLU epilogue to `matmul_kernel` behind a
   `RELU: tl.constexpr` flag, and test it against
   `torch.relu(a @ b)`. Why is a constexpr better than a run-time flag?

    <details markdown="1"><summary>Answer</summary>

    `if RELU: acc = tl.maximum(acc, 0.0)` before the store. As a constexpr,
    each value compiles its own kernel and the branch disappears; a run-time
    flag would keep a (uniform, cheap) branch in every program and prevent
    the compiler from specialising the epilogue.

    </details>

5. `m_safe` matters only when a row has seen nothing but masked keys.
   Show that this cannot happen in `flash_attention` as written, and name
   a variant of attention in which it does.

    <details markdown="1"><summary>Answer</summary>

    The first tile always starts at key 0, which is valid for every query
    row: it exists ($n \ge 1$), and it is not in the future of any row
    under the causal mask (padding rows past $n$ also see it). So after the
    first tile every `m` is finite. It fails in sliding-window attention
    (row $r$ sees only keys $r - w, \dots, r$, so the early tiles are fully
    masked for late rows), with key-padding masks, or when tiles are visited
    in a different order (e.g. starting at the diagonal). There `s - m_new`
    would be $-\infty - (-\infty) = \text{NaN}$ without the guard.

    </details>

## Practice

LeetGPU and Tensara accept Triton submissions; these problems are good
first ports of the kernels above:

- [LeetGPU – Vector Addition](../leetgpu/001-vector-add/)
- [LeetGPU – Softmax](../leetgpu/005-softmax/), [Tensara – Softmax](../tensara/softmax/)
- [LeetGPU – Matrix Multiplication](../leetgpu/002-matrix-multiplication/),
  [Tensara – Matrix Multiplication](../tensara/matrix-multiplication/)
- [LeetGPU – Softmax Attention](../leetgpu/006-softmax-attention/),
  [LeetGPU – Causal Attention](../leetgpu/053-casual-attention/)
- [Tensara – Layer Norm](../tensara/layer-norm/) (a fused reduction, like section 4)

# 01 – CUDA Execution Model

This chapter covers:

- how threads are organised (grid, block, warp) and how that maps onto the
  hardware;
- index arithmetic;
- why a GPU needs so many threads (latency hiding);
- occupancy, and how to pick a block size.

## 1. The Hierarchy

```
Grid  ── many Blocks  (scheduled independently onto SMs, in any order)
Block ── up to 1024 Threads (share shared memory, can __syncthreads())
Warp  ── 32 consecutive threads of a block, issued together (SIMT)
```

| Software | Hardware | Notes |
|---|---|---|
| Grid | The whole GPU | One kernel launch |
| Block (CTA) | One SM | A block never migrates; an SM can host several blocks at once |
| Warp | A warp scheduler slot | 32 threads that share one instruction stream |
| Thread | A lane of the SIMD units | Has its own registers and predicate |

![Blocks of a grid are placed on SMs; the warps of a block are issued by the SM's warp schedulers](figures/ch01-hierarchy.svg)

- A **kernel** is a function run by every thread of a grid.
- `blockIdx`, `blockDim`, `threadIdx` and `gridDim` are built-in `dim3`
  variables.
- Blocks cannot synchronize with each other inside a kernel (short of a
  cooperative launch). If you need a global barrier, end the kernel and
  launch another; kernels on the same stream run in order.
- Since Volta, each thread has its own program counter ("independent
  thread scheduling"). A warp still *issues* one instruction at a time for
  the lanes that are on the same path, so divergence still costs time,
  and warp-level primitives take an explicit lane mask (`0xffffffff`).

## 2. Index Arithmetic

A thread finds its element from its coordinates:

$$
i = b_x\,B_x + t_x, \qquad
\text{row} = b_y\,B_y + t_y, \quad \text{col} = b_x\,B_x + t_x, \qquad
G_x = \left\lceil \frac{n}{B_x} \right\rceil = \left\lfloor \frac{n + B_x - 1}{B_x} \right\rfloor
$$

| Symbol | Meaning |
|---|---|
| $t_x, t_y$ | `threadIdx.x`, `threadIdx.y`: position inside the block |
| $b_x, b_y$ | `blockIdx.x`, `blockIdx.y`: position of the block in the grid |
| $B_x, B_y$ | `blockDim.x`, `blockDim.y`: block size |
| $G_x$ | `gridDim.x`: number of blocks needed to cover $n$ elements |
| $i$ | Global 1-D index |
| Row, col | Global 2-D coordinates; `col` uses $x$ so that consecutive threads touch consecutive columns (coalescing, chapter 02) |

The last block is usually partial, so every thread must check its index:

```cpp
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

constexpr int kBlockSize = 256;
const int num_blocks = (n + kBlockSize - 1) / kBlockSize;
vectorAdd<<<num_blocks, kBlockSize>>>(d_a, d_b, d_c, n);
```

Inside a block, threads are numbered $x$-fastest, and warps are formed from
consecutive numbers:

$$
\tau = t_x + B_x\,(t_y + B_y\,t_z), \qquad w = \left\lfloor \frac{\tau}{32} \right\rfloor, \qquad \ell = \tau \bmod 32
$$

| Symbol | Meaning |
|---|---|
| $\tau$ | Linear thread index inside the block |
| $w$ | Warp index inside the block (`warp_id`) |
| $\ell$ | Lane index inside the warp (`lane`) |

So a $32\times8$ block has 8 warps, each one a full row of $t_x$ values.
A $16\times16$ block also has 8 warps, but each warp covers *two* rows.

![Left: the global row and column of a thread from its block and thread indices. Right: how warps are cut out of a 16 × 16 and a 32 × 8 block](figures/ch01-indexing.svg)

## 3. Grid-Stride Loops

Decouple the grid size from the problem size:

```cpp
__global__ void relu(const float* in, float* out, size_t n) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = fmaxf(in[i], 0.0f);
    }
}
```

Each thread handles

$$
k = \left\lceil \frac{n - i_0}{G_x B_x} \right\rceil \quad \text{elements}, \qquad i_0 = b_x B_x + t_x
$$

| Symbol | Meaning |
|---|---|
| $k$ | Iterations done by the thread whose first index is $i_0$ |
| $G_x B_x$ | Total number of threads, the loop stride |

Benefits:

- any $n$ works, including $n > 2^{31}$ (use `size_t`);
- the grid can be sized to the machine (a few waves of blocks per SM)
  rather than to the data;
- per-thread setup cost (loading a constant, initialising an
  accumulator) is amortised over $k$ elements, which is how reductions
  build their per-thread partial sums (chapter 03).

## 4. Warps and Divergence

All 32 lanes of a warp execute the same instruction. When lanes take
different branches, the warp runs each path in turn with the other lanes
masked off:

$$
T_{\text{warp}} \approx \sum_{p \in \text{paths taken}} T_p
$$

| Symbol | Meaning |
|---|---|
| $p$ | A distinct control-flow path taken by at least one lane |
| $T_p$ | Time to execute path $p$ |

- A branch that is **uniform across the warp** (`if (warp_id == 0)`,
  `if (blockIdx.x < k)`) costs nothing extra.
- Short branches (`x > 0 ? x : a * x`) compile to predicated selects; no
  real divergence.
- Loops with lane-dependent trip counts run as long as the longest lane.

## 5. Why So Many Threads: Latency Hiding

A global load takes hundreds of cycles. The GPU does not wait; the warp
scheduler switches to another warp that is ready. Little's law says how
much work must be in flight to keep the memory system busy:

$$
N_{\text{bytes in flight}} = \beta \times L, \qquad
N_{\text{warps}} \gtrsim \frac{\beta\,L}{n_{\text{SM}}\cdot b_{\text{warp}}}
$$

| Symbol | Meaning |
|---|---|
| $\beta$ | DRAM bandwidth (bytes/s) |
| $L$ | Memory latency (s) |
| $N_{\text{bytes in flight}}$ | Bytes that must be requested but not yet returned, at any moment |
| $n_{\text{SM}}$ | Number of SMs |
| $b_{\text{warp}}$ | Bytes in flight per warp (for example 512 for one `float4` load per lane) |
| $N_{\text{warps}}$ | Resident warps needed per SM |

![While one warp waits for memory, the scheduler issues instructions from other warps](figures/ch01-latency-hiding.svg)

For an A100 ($\beta \approx 1.5$ TB/s, $L \approx 500$ ns, 108 SMs):
$\beta L \approx 750$ KB, i.e. about 7 KB per SM, or ~14 warps per SM each
with one 512-byte load outstanding. Fewer warps are fine if each has more
independent loads in flight: unrolled loops and `float4` loads raise
$b_{\text{warp}}$. This is the **instruction-level parallelism**
alternative to occupancy.

## 6. Occupancy

An SM runs as many blocks as its resources allow:

$$
n_{\text{blocks/SM}} = \min\left(
\left\lfloor \frac{T_{\max}}{B} \right\rfloor,\
\left\lfloor \frac{R_{\text{SM}}}{r\,B} \right\rfloor,\
\left\lfloor \frac{S_{\text{SM}}}{s} \right\rfloor,\
n_{\max}\right), \qquad
\text{occupancy} = \frac{n_{\text{blocks/SM}}\cdot B}{T_{\max}}
$$

| Symbol | Meaning |
|---|---|
| $B$ | Threads per block |
| $T_{\max}$ | Maximum resident threads per SM (2048 on A100/H100) |
| $R_{\text{SM}}$ | Registers per SM (65 536) |
| $r$ | Registers per thread (from `-Xptxas -v`; allocated in chunks) |
| $S_{\text{SM}}$ | Shared memory per SM available to blocks (up to ~164 KB on A100, ~228 KB on H100) |
| $s$ | Shared memory per block (static plus dynamic) |
| $n_{\max}$ | Maximum resident blocks per SM (32) |
| Occupancy | Fraction of the SM's thread slots in use |

Example: $B = 256$, $r = 64$, $s = 32$ KB on an A100 gives
$\min(8, 4, 5, 32) = 4$ blocks, i.e. 1024 threads and 50 % occupancy.
Registers are the limit; `__launch_bounds__(256, 6)` would ask the
compiler to use at most 40 registers (at the risk of spills).

![Each resource limits the number of resident blocks; the smallest limit sets the occupancy](figures/ch01-occupancy.svg)

Occupancy is a means, not a goal. Register-blocked GEMMs run very well at
12–25 % occupancy because each thread has many independent FMAs and loads
in flight. Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` or the
Nsight Compute occupancy section to see the limiter.

## 7. Choosing a Block Size

- A multiple of 32. 128–512 is typical; 256 is a safe default.
- For 2-D problems, make the $x$ extent at least 32 so that each warp
  covers one contiguous row segment (a $32\times8$ block).
- For block-wide reductions, larger blocks mean fewer partial results
  but a longer tree; 256–1024 is common.
- Launch at least a few blocks per SM ($\ge 2 n_{\text{SM}}$), or the
  tail of the last wave leaves SMs idle. With $G$ blocks and $c$ blocks
  resident per SM, the number of waves is
  $\lceil G / (c\,n_{\text{SM}}) \rceil$; a grid of 1.1 waves wastes almost
  half of the second wave.

## 8. Synchronization and Communication

| Scope | Mechanism |
|---|---|
| Warp | `__shfl_*_sync`, `__ballot_sync`, `__syncwarp()` |
| Block | Shared memory + `__syncthreads()` |
| Grid | Kernel boundary; atomics (`atomicAdd`, `atomicMax`, …); cooperative groups `grid.sync()` with a cooperative launch |
| Host | `cudaDeviceSynchronize()`, events, stream ordering |

`__syncthreads()` must be reached by **every** thread of the block. A
barrier inside `if (threadIdx.x < 16)` deadlocks or corrupts data.

## 9. Worked Example: Vectorized Vector Addition

```cpp
constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 4096;

__global__ void vectorAddVec4(const float* a, const float* b, float* c, size_t n) {
    const size_t num_vec4 = n / 4;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* c4 = reinterpret_cast<float4*>(c);
    for (size_t i = start; i < num_vec4; i += stride) {
        const float4 x = a4[i], y = b4[i];
        c4[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
    for (size_t i = num_vec4 * 4 + start; i < n; i += stride) c[i] = a[i] + b[i];   // tail
}
```

It combines everything above: a grid-stride loop (any $n$), 16-byte
accesses (more bytes in flight per warp), a capped grid, and 64-bit
indices. The full solution with its cost analysis is
[Tensara – Vector Addition](../tensara/vector-addition/).

## Practice

- [LeetGPU – Vector Addition](../leetgpu/001-vector-add/)
- [Tensara – Vector Addition](../tensara/vector-addition/)
- [LeetGPU – Matrix Addition](../leetgpu/008-matrix-addition/) (2-D indexing)
- [LeetGPU – Reverse Array](../leetgpu/019-reverse-array/) (in-place, half the threads)

# 02 – Memory Hierarchy and Coalescing

Most kernels in the practice sets are **memory-bound**: their speed is set
by how many bytes they move and how efficiently they move them. This
chapter covers:

- the memory spaces and what each is for;
- coalescing: how a warp's accesses become DRAM transactions;
- shared-memory banks and conflicts;
- a fully worked transpose;
- how to measure the result.

## 1. The Memory Spaces

| Memory | Scope | Latency (approx.) | Size (A100, approx.) | Notes |
|--------|-------|-------------------|------|-------|
| Registers | thread | ~1 cycle | 256 KB per SM | Spills go to "local" memory (slow, cached in L1/L2). |
| Shared memory | block | ~20–30 cycles | up to 164 KB per SM | Programmer-managed, 32 banks. Shares storage with L1. |
| L1 cache | SM | ~30 cycles | 192 KB per SM (incl. shared) | Automatic; caches global loads. |
| L2 cache | device | ~200 cycles | 40 MB | Automatic; all SMs share it; atomics resolve here. |
| Global (HBM / GDDR) | device | ~400–800 cycles | 40–80 GB | Large, high bandwidth, high latency. |
| Constant | device, read-only | cached | 64 KB | Fast when all lanes read the same address (broadcast). |

The numbers vary between generations. The ratios are what matters: each
level down is roughly an order of magnitude slower, and DRAM bandwidth is
roughly 10–20× lower than the rate at which the SMs can do arithmetic.

## 2. Coalescing

A warp's global load is split into **32-byte sectors** (four sectors form
a 128-byte cache line). The hardware fetches every sector that at least
one lane touches. For a warp in which lane $\ell$ reads $e$ bytes at
address $a_0 + \ell\,s\,e$:

$$
n_{\text{sectors}} \approx \min\left(32,\ \left\lceil \frac{32\,s\,e}{32} \right\rceil\right) \ \ (\text{aligned } a_0), \qquad
\eta = \frac{32\,e}{32\,n_{\text{sectors}}}
$$

| Symbol | Meaning |
|---|---|
| $\ell$ | lane index, $0 \dots 31$ |
| $e$ | bytes per lane (4 for `float`, 16 for `float4`) |
| $s$ | stride between consecutive lanes, in elements |
| $a_0$ | address read by lane 0 |
| $n_{\text{sectors}}$ | 32-byte sectors fetched for the whole warp |
| $\eta$ | efficiency: useful bytes over fetched bytes |

| Pattern | $s$ | sectors | $\eta$ |
|---|---|---|---|
| `x[i]`, `float` | 1 | 4 | 100 % |
| `x[i]`, `float4` | 1 | 16 | 100 % (and 4× fewer instructions) |
| `x[2 * i]`, `float` | 2 | 8 | 50 % |
| `x[32 * i]`, `float` (a column of a 32-wide matrix) | 32 | 32 | 12.5 % |
| misaligned by 4 bytes, `float` | 1 | 5 | 80 % |

**Rule of thumb:** make `threadIdx.x` index the fastest-varying
(contiguous) dimension. When a kernel must read along the slow dimension
(a transpose, a column reduction), either let *neighbouring threads* take
neighbouring columns so that each warp access is still contiguous (see
[Tensara – Argmax](../tensara/argmax/)), or stage the data through shared
memory.

Interleaved layouts are a special case: 32 lanes reading `rgb[3 * i]`
touch 12 sectors for 128 useful bytes, but the next two instructions
(`rgb[3 * i + 1]`, `rgb[3 * i + 2]`) hit the same lines in L1, so DRAM
traffic is still optimal ([Tensara – Grayscale](../tensara/grayscale/)).

## 3. Vectorized Access

`float4` (or `int4`, `uint2`, …) loads move 16 bytes per lane per
instruction:

- fewer load and store instructions per byte;
- more bytes in flight per warp, which helps latency hiding (chapter 01);
- requires 16-byte alignment. `cudaMalloc` returns 256-byte-aligned
  pointers; a row of an $M\times K$ matrix is aligned only if $K$ is a
  multiple of 4.

```cpp
const float4 v = reinterpret_cast<const float4*>(in)[i];   // i indexes float4s
```

## 4. Shared Memory and Bank Conflicts

Shared memory is split into 32 **banks** of 4 bytes. Successive 4-byte
words go to successive banks:

$$
\operatorname{bank}(a) = \left\lfloor \frac{a}{4} \right\rfloor \bmod 32, \qquad
\text{degree} = \max_{k}\ \bigl\lvert \{\text{distinct words in bank } k \text{ requested by the warp}\} \bigr\rvert
$$

| Symbol | Meaning |
|---|---|
| $a$ | byte address in shared memory |
| $\operatorname{bank}(a)$ | bank that serves the address |
| degree | conflict degree: the access is split into this many serial transactions |

Lanes that read the **same word** do not conflict (broadcast). Lanes that
read **different words of the same bank** do.

Reading a column of a `float tile[32][32]` is the worst case: element
$(r, c)$ is at word $32r + c$, so the 32 lanes (different $r$, same $c$) all
hit bank $c$, a 32-way conflict. Padding each row by one word fixes it:

$$
\operatorname{bank}\bigl(\text{tile}[r][c]\bigr) = \bigl(r\,(T + p) + c\bigr) \bmod 32
\ \xrightarrow{\ T = 32,\ p = 1\ }\ (r + c) \bmod 32
$$

| Symbol | Meaning |
|---|---|
| $T$ | tile width (32) |
| $p$ | padding words per row |
| $r, c$ | row and column in the tile |

For fixed $c$ and $r = 0 \dots 31$, $(r + c) \bmod 32$ takes 32 distinct
values: conflict-free.

```cpp
__shared__ float tile[kTile][kTile + 1];   // +1 shifts each row by one bank
```

Other fixes are an XOR swizzle of the column index (used by CUTLASS and
the AMD kernels of chapters 06–07), or choosing which dimension
`threadIdx.x` indexes so that lanes read along a row.

## 5. Worked Example: A Coalesced Transpose

$B = A^{\mathsf T}$ for an $R\times C$ matrix. A direct kernel reads rows
and writes columns (or vice versa), so one side is uncoalesced. Tiling
through shared memory makes both sides contiguous:

```cpp
constexpr int kTile = 32;
constexpr int kRowsPerPass = 8;

// launch: block(kTile, kRowsPerPass), grid(ceil(cols / kTile), ceil(rows / kTile))
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[kTile][kTile + 1];
    int x = blockIdx.x * kTile + threadIdx.x;              // input column
    for (int dy = threadIdx.y; dy < kTile; dy += kRowsPerPass) {
        const int y = blockIdx.y * kTile + dy;             // input row
        if (x < cols && y < rows) tile[dy][threadIdx.x] = in[static_cast<size_t>(y) * cols + x];
    }
    __syncthreads();
    // Swap the block coordinates so that the write is coalesced too.
    x = blockIdx.y * kTile + threadIdx.x;                  // output column = input row
    for (int dy = threadIdx.y; dy < kTile; dy += kRowsPerPass) {
        const int y = blockIdx.x * kTile + dy;             // output row = input column
        if (x < rows && y < cols) out[static_cast<size_t>(y) * rows + x] = tile[threadIdx.x][dy];
    }
}
```

- **Loads**: lanes read 32 consecutive floats of one input row
  (4 sectors, $\eta = 100\%$). The shared store `tile[dy][threadIdx.x]`
  goes along a row: conflict-free.
- **Stores**: lanes write 32 consecutive floats of one output row. The
  shared read `tile[threadIdx.x][dy]` goes down a column; thanks to the
  padding it is conflict-free.
- **Each thread handles 4 elements** ($32\times8$ threads for a
  $32\times32$ tile), which amortises index math and gives each warp
  several independent loads in flight.

$$
Q = 2 \cdot 4RC\ \text{bytes}, \qquad T_{\min} = \frac{8RC}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $R, C$ | rows and columns of the input |
| $Q$ | compulsory DRAM traffic: read every element once, write it once |
| $\beta$ | DRAM bandwidth |

A good transpose reaches 80–90 % of the bandwidth of a plain copy. A naive
one (reads coalesced, writes strided by $R$) moves up to 8× more sectors
on the write side and is typically 3–5× slower.

## 6. Caches and Read-Only Data

- `const T* __restrict__` tells the compiler that the data is not
  written through another pointer during the kernel, which lets it use
  the read-only (non-coherent) path and reorder loads freely.
- `__ldg(p)` forces that path explicitly.
- Data read by every thread with the same address in the same instruction
  (a filter kernel, a bias) should be in `__constant__` memory or a
  register-cached broadcast.
- L2 is large (40–50 MB on current data-centre GPUs). A tensor read
  twice in quick succession (a row re-read in the second pass of a
  normalization) often comes from L2 at several times DRAM speed, which
  is why "two-pass" kernels are cheaper than their byte count suggests.

## 7. Measuring

Effective bandwidth, as in chapter 00:

$$
\beta_{\text{eff}} = \frac{Q_{\text{read}} + Q_{\text{written}}}{t}, \qquad
\text{efficiency} = \frac{\beta_{\text{eff}}}{\beta_{\text{peak}}}
$$

| Symbol | Meaning |
|---|---|
| $Q_{\text{read}}, Q_{\text{written}}$ | compulsory bytes (not counting cache re-reads) |
| $t$ | kernel time |
| $\beta_{\text{peak}}$ | datasheet bandwidth, e.g. ~1.55 TB/s on A100 40 GB, ~320 GB/s on T4 |

In Nsight Compute, the *Memory Workload Analysis* section reports DRAM
throughput, L1/L2 hit rates and "sectors per request" (4 is ideal for
`float`, 16 for `float4`). The *Source* view lists the instructions with
uncoalesced accesses and shared-memory bank conflicts.

## Practice

- [LeetGPU – Matrix Transpose](../leetgpu/003-matrix-transpose/)
- [LeetGPU – Matrix Copy](../leetgpu/031-matrix-copy/)
- [Tensara – Grayscale](../tensara/grayscale/) (interleaved layout)
- [Tensara – Max Dim](../tensara/max-dim/) (strided reduction, coalesced across threads)

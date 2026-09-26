# 02 – Memory Hierarchy & Coalescing

| Memory | Scope | Latency (approx.) | Notes |
|--------|-------|-------------------|-------|
| Registers | thread | ~1 cycle | Spills go to "local" memory (slow). |
| Shared memory | block | ~20–30 cycles | Programmer-managed cache, 32 banks. |
| L1 / L2 cache | SM / device | ~30 / ~200 cycles | Automatic. |
| Global (HBM / GDDR) | device | ~400–800 cycles | Large, high bandwidth, high latency. |
| Constant | device, read-only | cached | Fast when all threads read the same address. |

## Coalescing

A warp's global memory request is served in 32-byte sectors. When the 32
threads read 32 consecutive `float`s, that's 4 sectors — ideal. Strided access
(`a[threadIdx.x * stride]`) touches up to 32 sectors for the same useful data.

**Rule of thumb:** make `threadIdx.x` index the fastest-varying (contiguous)
dimension.

## Shared memory & bank conflicts

Shared memory is split into 32 banks of 4 bytes. If two threads of a warp hit
different addresses in the same bank, the accesses serialize.

Classic fix in a transpose: pad the tile.

```cpp
__shared__ float tile[kTile][kTile + 1];   // +1 shifts each row by one bank
```

## Example: coalesced transpose

```cpp
constexpr int kTile = 32;

__global__ void transpose(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[kTile][kTile + 1];
    int x = blockIdx.x * kTile + threadIdx.x;
    int y = blockIdx.y * kTile + threadIdx.y;
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    __syncthreads();
    // Swap block coordinates so the write is also coalesced.
    x = blockIdx.y * kTile + threadIdx.x;
    y = blockIdx.x * kTile + threadIdx.y;
    if (x < rows && y < cols) out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}
// launch: block(kTile, kTile), grid(ceil(cols / kTile), ceil(rows / kTile))
```

## Vectorized access

`float4` loads move 16 bytes per instruction. Requires 16-byte alignment.
See [Tensara – Vector Addition](../tensara/vector-addition).

## Measuring

Effective bandwidth = `(bytes_read + bytes_written) / time`. Compare to the
GPU's peak (e.g. ~2 TB/s on A100 40GB, ~320 GB/s on T4).
